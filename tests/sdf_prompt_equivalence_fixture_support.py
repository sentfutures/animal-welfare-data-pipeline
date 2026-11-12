"""Engine for the SDF prompt-equivalence test and the final corpus schema test.

The pipeline is run end to end through ``sdf_pipeline/run.py``'s real ``main`` on small fixtures
(the scenarios, defined in ``tests/sdf_prompt_equivalence_scenarios.py``) with canned replies, and
every request that would reach a model provider is recorded exactly as it would have been sent.
``tests/test_sdf_prompt_equivalence_and_final_corpus.py`` compares the recordings with files
committed under ``tests/fixtures/sdf_prompt_equivalence/`` (one per scenario), which were recorded
on ``origin/main``; ``scripts/regenerate_sdf_prompt_equivalence_recorded_requests.py`` is the only
code that rewrites them.

How requests are recorded
-------------------------
Only two classes are replaced between the pipeline layers and the provider: ``anthropic.Anthropic``
and ``openai.OpenAI`` become fakes whose ``messages.create`` / ``responses.create`` record the
keyword arguments they are given and return a canned reply. Everything above them runs for real:
``shared.api.call_claude`` (model resolution, thinking headroom, reasoning effort from the stage
tag, temperature), the real ``shared.api._call_with_retry`` (which adds ``thinking`` and
``output_config`` and drops the temperature for models that do not accept it),
``shared.openai_api.call_openai`` and the real ``_create_response``. What is recorded is what
reaches the wire, including client constructor settings (timeout, retries, default headers; keys
are not recorded, only that one was supplied).

``call_claude`` and ``call_openai`` are wrapped only to note the stage tag and item id of the call
in a thread-local variable, because those never reach the wire and the canned replies are chosen by
them. Replies are chosen by request, never by queue: the key is (stage tag, item id, attempt), the
attempt being the number of requests already sent for that stage and item in the scenario (across
resumes, which are further invocations of ``main`` in the same scenario).

What else is recorded per scenario and invocation
-------------------------------------------------
* nothing that depends on installed library data: the name pools the composer hands each plan
  prompt come from ``FIXED_ENTITY_POOLS_BY_LOCALE`` below, not from Faker, whose locale data differs
  between versions (requirements.txt leaves faker unpinned);
* the configuration used (so a change to the real ``config.yaml`` is reported as such);
* the exit message of each invocation (stops are ``SystemExit`` messages);
* every stage file under layer1..layer4 (field names, field order, values) and the final corpus.
  Normalised, and only this: checkpoint files' ``completed`` list is sorted and ``last_updated``
  replaced; the two files appended to in thread-completion order (``consideration_scores.jsonl``
  and ``consideration_scoring_failures.jsonl``) are sorted by ``prompt_id``; the run directory's
  absolute path is replaced by ``<run directory>``.

How the recorded files stay small
---------------------------------
The recorded files hold hashes for long text, never the text itself, so that they stay small enough
to review. Every message text of a request (Anthropic ``messages`` content, OpenAI ``input``) is
stored as its sha256 and character count whatever its length, as system prompts are; every other
request field (model, max tokens, effort, temperature, schema, roles, structure) stays in plain
text and is compared exactly. In stage files, checkpoints and the final corpus, a string longer than
``LONG_VALUE_CHARACTER_THRESHOLD`` is stored as its sha256 and length; field names, field order,
record order and every shorter value stay in plain text. The recorder and the outcome keep the
full texts in memory: only ``hashed_form_of_record`` turns them into hashes, and when a hash does
not match, the comparison reports the new text (see ``describe_changed_hashed_text``), so a failure
shows what is now being sent or written.

Each dealt hand (a record's ``variables`` dictionary) appears in many stage files, checkpoints and
final corpus records, so a recorded file holds each distinct hand once, in ``dealt_hands_by_identifier``,
and every record stores ``"dealt_hand:<identifier>"`` where the dictionary was. The identifier is the
first 16 hex digits of the sha256 of the hand's JSON with its keys in order, so it is determined by the
hand alone (a prompt id is not: a redeal puts other cards under a later hand's id, and a resume
rebuilds hands). ``load_recorded_scenario`` expands the references again, so the comparison sees
the full hand in each record, in its recorded position and key order, and a changed card is
reported as ``record.variables.<variable>: recorded <old>, now <new>``.

The same is done for every other repeated structure: a dictionary or list whose compact JSON is at
least ``MINIMUM_CHARACTERS_OF_REPEATED_VALUE_TO_STORE_ONCE`` long and that occurs more than once in
the recorded content (a consideration score record or coverage table that a resume leaves unchanged,
the configuration of each invocation, a request's system block) is stored once in
``repeated_values_by_identifier`` and referred to as ``"repeated_value:<identifier>"``. Loading
expands both kinds of reference, so a difference is still reported at its full field path.
"""

import hashlib
import inspect
import json
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import yaml

from shared import api, api_keys, entity_pools, openai_api, utils
from sdf_pipeline import considerations_catalog
import sdf_pipeline.run as sdf_run

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests" / "fixtures" / "sdf_prompt_equivalence"
RECORDED_FILE_FORMAT_VERSION = 4   # 3: message texts and long values stored as hash + length; 4: dealt hands and other repeated values stored once per file
REGENERATION_COMMAND = "python scripts/regenerate_sdf_prompt_equivalence_recorded_requests.py"
REGENERATION_RULE = (
    f"The recorded files are rewritten only by `{REGENERATION_COMMAND}`, in the same pull request "
    "as the intended request change; a pull request that only refactors never rewrites them.")
# Strings in stage files, checkpoints and the final corpus longer than this are stored as a hash and
# a length. 400 keeps every id, score, label, flag and dealt variable value readable (the longest
# value any variables.txt line renders to is under 400 characters, so a dealt hand reads in the
# recorded file) while hashing document texts, planned and rewritten descriptions, plans, critiques
# and similar prose, which are what make the files large.
LONG_VALUE_CHARACTER_THRESHOLD = 400
CHARACTERS_OF_CHANGED_TEXT_SHOWN_IN_FAILURE = 2000
# A request's JSON schema whose compact JSON (keys in generation order) is longer than this is stored
# as its hash and length; a smaller one stays in plain text. The big ones are the enumerations of the
# screen, selection and rewrite schemas, which make up a third of the request data.
LARGE_SCHEMA_CHARACTER_THRESHOLD = 1500
MAXIMUM_CHARACTERS_OF_ONE_LINE_CONTAINER = 200


def recorded_path_for_scenario(scenario_name: str) -> Path:
    return FIXTURE_DIRECTORY / f"recorded_{scenario_name}_on_origin_main.json"


# The real network-seam functions, captured when this module is first imported (at test
# collection), before tests/conftest.py's autouse guards replace them with functions that raise.
REAL_ANTHROPIC_CALL_WITH_RETRY = api._call_with_retry
REAL_OPENAI_CREATE_RESPONSE = openai_api._create_response
assert REAL_ANTHROPIC_CALL_WITH_RETRY.__name__ != "_blocked", (
    "tests/sdf_prompt_equivalence_fixture_support.py was imported after the conftest guard "
    "replaced shared.api._call_with_retry; import it at module level of a test file")

STAGE_TAGS_OF_SDF_PIPELINE = (
    "layer1_screen", "layer1_plan", "layer1_description_rewrite", "layer1_select",
    "layer2_draft", "layer3_critique", "layer3_revise", "layer3_rewrite", "layer4_score",
)

# One distinct reasoning effort per stage tag, so a renamed or mistyped tag changes the request.
# (The committed config.yaml sets no effort for the SDF stage tags; the fixture config sets them
# all so that the lookup from stage tag to effort is observable. The real-config scenario does not.)
REASONING_EFFORT_BY_STAGE_TAG = {
    "layer1_screen": "low",
    "layer1_plan": "medium",
    "layer1_description_rewrite": "high",
    "layer1_select": "xhigh",
    "layer2_draft": "max",
    "layer3_critique": "low",
    "layer3_revise": "medium",
    "layer3_rewrite": "high",
    "layer4_score": "xhigh",
}


# ---------------------------------------------------------------------------------- fixed name pools

# ``shared.entity_pools.build_pools_for_locale`` builds locale-matched name pools from seeded Faker,
# whose locale data changes between releases, so the names in recorded plan and draft prompts would
# change with the installed Faker and fail the test with no pipeline change. Every scenario replaces
# that function with one returning these committed pools, one per locale ``entity_pools`` can map a
# culture to (a culture mapped to None never reaches the function: the composer uses its
# instruction-only fallback). The names are invented for the fixture. The pipeline's own seeded
# sampling from a pool (``entity_pools.sample_for``) still runs for real.
def _fixed_pool(people: str, organisations: str) -> tuple[list[str], list[str]]:
    return [name.strip() for name in people.split(";")], [name.strip() for name in organisations.split(";")]


FIXED_ENTITY_POOLS_BY_LOCALE = {
    "en_US": _fixed_pool("Dana Whitlock; Marcus Aldrin Pryce; Joelle Hartnett; Tobias Renner; Imani Calloway; Wendell Okafor-Reyes; Priscilla Dunmore; Anders Kell",
                         "Harlow Ridge Cooperative; Pinecrest Analytics; Bellwether Foods; Stonegate Veterinary Partners; Orchard Lane Press"),
    "en_GB": _fixed_pool("Rhiannon Thackeray; Alistair Penhallow; Bethan Corrigan; Hamish Duffield; Philippa Lowndes; Ewan Garside; Nadine Ashcroft; Cedric Fenwick",
                         "Tamar Valley Trust; Wrexall and Daughters; Kestrel Fen Partnership; Longmead Group; Ashdown Press"),
    "en_IN": _fixed_pool("Anjali Deshmukh; Rohit Venkataraman; Meenakshi Pillai; Sanjay Kulkarni; Farhana Siddiqui; Gurpreet Dhillon; Lakshmi Narasimhan; Imran Qureshi",
                         "Kaveri Basin Society; Sundaram Agro Works; Himalaya Commons Trust; Lotus Delta Traders; Neem Grove Cooperative"),
    "fr_FR": _fixed_pool("Margaux Delacroix; Thibault Lenormand; Soline Brunet; Gaspard Vidal; Clémence Rochefort; Anselme Bourdin; Odile Marchetti; Lucien Faure",
                         "Coopérative des Hautes Terres; Maison Valmont; Atelier Rivière et Fils; Groupe Saint-Aubin; Institut du Bocage"),
    "de_DE": _fixed_pool("Annelie Brandt; Konstantin Falkenrath; Mareike Sonntag; Dietmar Wolters; Hanna Eichborn; Lutz Hartmann-Voss; Ottilie Kranz; Jannik Pfeiffer",
                         "Genossenschaft Weserland; Hartmann und Söhne; Institut für Tierhaltung Nord; Falkenrath GmbH; Stiftung Moorwiese"),
    "es_ES": _fixed_pool("Itziar Montalbán; Gonzalo Arteaga; Remedios Caballero; Iker Zubiaur; Macarena Olmedo; Facundo Benavides; Lorena Quintanilla; Jacinto Roldán",
                         "Cooperativa Sierra Clara; Bodegas Almendral; Instituto del Ebro; Grupo Ventura y Hermanos; Fundación Dehesa Viva"),
    "es_MX": _fixed_pool("Xóchitl Barragán; Aurelio Mondragón; Citlali Ibarra; Evaristo Peñaloza; Itzel Cordero; Baltazar Ugalde; Marisol Treviño; Cuauhtémoc Lagunas",
                         "Cooperativa Milpa Alta; Grupo Tlaloc; Instituto de la Sierra Madre; Hermanos Olvera e Hijos; Fundación Cenote"),
    "it_IT": _fixed_pool("Ludovica Barbieri; Tancredi Sartori; Fiammetta Colombo; Evaristo Mancuso; Ginevra Palumbo; Ottavio Ferraiolo; Serena Cattaneo; Lorenzo Bellandi",
                         "Cooperativa Colle Verde; Azienda Fratelli Moretti; Istituto del Po; Gruppo Santoro; Fondazione Uliveto"),
    "pt_BR": _fixed_pool("Larissa Cavalcanti; Heitor Nogueira; Valentina Albuquerque; Otávio Sampaio; Bianca Figueiredo; Murilo Tavares; Cecília Drummond; Raimundo Paes",
                         "Cooperativa Várzea Grande; Irmãos Vasconcelos; Instituto do Cerrado; Grupo Maré Alta; Fundação Caatinga"),
    "nl_NL": _fixed_pool("Sanne Verbeek; Joost Vanderhoek; Marit Kuipers; Bram Oostrom; Eline Hoekstra; Wouter Brinkman; Floor Zwanenburg; Pieter Dekkers",
                         "Coöperatie Polderlicht; Van der Meer en Zonen; Instituut voor Veehouderij; Groep Waddenzee; Stichting Veenweide"),
    "pl_PL": _fixed_pool("Wiktoria Zielińska; Radosław Kamiński; Aleksandra Dąbrowska; Bogdan Wróblewski; Marta Szczepańska; Lech Sokołowski; Ewelina Jankowska; Mieczysław Piotrowski",
                         "Spółdzielnia Kurpie; Zakłady Mazurskie; Instytut Nizinny; Grupa Wisła; Fundacja Puszcza"),
    "sv_SE": _fixed_pool("Linnea Strömberg; Gustav Wallenberg-Holm; Ingrid Nordlund; Sixten Åkerlund; Maja Hedlund; Torsten Lindqvist; Elsa Bergkvist; Olof Sandberg",
                         "Kooperativet Norrskog; Åkerlund och Söner; Institutet för Havsmiljö; Grupp Mälardalen; Stiftelsen Fjällräven"),
    "cs_CZ": _fixed_pool("Kristýna Veselá; Ondřej Havlíček; Markéta Svobodová; Zdeněk Procházka; Lucie Dvořáková; Vladimír Kolář; Eliška Marešová; Jaroslav Němec",
                         "Družstvo Šumava; Továrna Havlík a syn; Ústav pro krajinu; Skupina Vltava; Nadace Lužické hory"),
    "ro_RO": _fixed_pool("Ioana Mărgineanu; Cătălin Ardeleanu; Simona Păun; Radu Tănase; Alexandra Georgescu; Vasile Moldovan; Elena Rădulescu; Dorin Stoica",
                         "Cooperativa Carpați; Fabrica Dumitrescu și Fiii; Institutul Dunărea de Jos; Grupul Banat; Fundația Delta Verde"),
    "tr_TR": _fixed_pool("Elif Karaca; Mehmet Aksoy; Zeynep Tuncer; Burak Yıldırım; Aylin Demirtaş; Kerem Özkan; Selin Çakır; Hasan Erdoğan",
                         "Anadolu Tarım Kooperatifi; Kaya ve Oğulları; Ege Hayvancılık Enstitüsü; Boğaz Grubu; Mavi Vadi Vakfı"),
    "id_ID": _fixed_pool("Ayu Wulandari; Budi Santoso Hartono; Citra Anggraini; Dimas Prakoso; Fitri Handayani; Gilang Permana; Intan Maharani; Joko Widodo Putra",
                         "Koperasi Tani Makmur; PT Sinar Nusantara; Institut Perikanan Selatan; Kelompok Kali Ciliwung; Yayasan Rimba Hijau"),
    "vi_VN": _fixed_pool("Nguyễn Thị Lan Anh; Trần Quốc Bảo; Lê Minh Châu; Phạm Gia Huy; Hoàng Thu Hà; Đặng Văn Khoa; Vũ Ngọc Mai; Bùi Thanh Tùng",
                         "Hợp tác xã Sông Hồng; Công ty Hải Âu; Viện Nghiên cứu Đồng bằng; Nhóm Mê Kông; Quỹ Rừng Xanh"),
    "no_NO": _fixed_pool("Ingvild Haugland; Sverre Aasen; Marte Fjeldstad; Tormod Bakke; Synnøve Lunde; Eirik Strand; Hilde Solbakken; Knut Myhre",
                         "Samvirket Fjordland; Aasen og Sønner; Institutt for Havbruk; Gruppen Nordkapp; Stiftelsen Vidda"),
    "hi_IN": _fixed_pool("अनन्या मिश्रा; विक्रम सिंह राठौर; कविता नायर; रजनीश तिवारी; सुनीता गुप्ता; मोहन कृष्णन; पूजा देशपांडे; अमित वर्मा",
                         "गंगा घाटी सहकारी समिति; शर्मा एंड संस; हिमालय अध्ययन संस्थान; विंध्य समूह; नर्मदा न्यास"),
    "bn_BD": _fixed_pool("তানভীর আহমেদ; সুমাইয়া রহমান; রফিকুল ইসলাম; নুসরাত জাহান; মাহফুজ হাসান; শারমিন আক্তার; কামরুল হোসেন; ফারহানা চৌধুরী",
                         "পদ্মা সমবায় সমিতি; করিম অ্যান্ড সন্স; বঙ্গোপসাগর গবেষণা কেন্দ্র; সুন্দরবন গ্রুপ; মেঘনা ফাউন্ডেশন"),
    "zh_CN": _fixed_pool("王晓梅; 李志远; 张雨涵; 刘建国; 陈思琪; 杨国强; 赵文静; 周明轩",
                         "长江流域合作社; 华远农业有限公司; 黄河生态研究所; 青山集团; 云岭基金会"),
    "ja_JP": _fixed_pool("佐藤美咲; 高橋健太; 伊藤彩花; 渡辺雄大; 小林真由; 中村拓海; 山本奈々; 加藤隆之",
                         "北上川流域協同組合; 白樺食品株式会社; 瀬戸内海洋研究所; 緑ヶ丘グループ; 山里財団"),
    "ko_KR": _fixed_pool("김서연; 박지훈; 이하은; 최민준; 정유진; 강도윤; 조수빈; 윤재현",
                         "낙동강유역협동조합; 한빛식품 주식회사; 서해바다연구소; 푸른들 그룹; 지리산재단"),
    "en_NG": _fixed_pool("Chidinma Okoro; Babatunde Adeyemi; Ngozi Eze; Tunde Bakare; Amina Yusuf; Emeka Nwosu; Folake Ogunleye; Ibrahim Danjuma",
                         "Niger Delta Growers Cooperative; Adeyemi and Sons; Lagos Lagoon Research Institute; Savanna Belt Group; Oyo Heritage Foundation"),
    "ar_SA": _fixed_pool("نورة العتيبي; فهد الدوسري; ريم الشمري; سلطان القحطاني; هند الحربي; ماجد الزهراني; لمى السبيعي; خالد المطيري",
                         "جمعية واحة الجنوب التعاونية; مؤسسة الرمال; معهد أبحاث البحر الأحمر; مجموعة الخليج; مؤسسة النخيل"),
}


def _fixed_build_pools_for_locale(locale, n_people=120, n_orgs=80, seed=137):
    return FIXED_ENTITY_POOLS_BY_LOCALE[locale]


def install_fixed_entity_pools(monkeypatch) -> None:
    """Make the composer's name pools independent of the installed Faker (see above)."""
    monkeypatch.setattr(entity_pools, "build_pools_for_locale", _fixed_build_pools_for_locale)


# ---------------------------------------------------------------------------------- scenarios

@dataclass(frozen=True)
class Invocation:
    """One call of ``sdf_pipeline/run.py``'s ``main``."""
    arguments: tuple = ()                          # extra command-line arguments (--resume, --layer N)
    before: Callable | None = None                 # given the run directory; edits files as an interruption would
    sdf_config_overrides: dict = field(default_factory=dict)   # for this invocation only
    expected_exit_message_fragment: str | None = None          # None: main must return normally


@dataclass(frozen=True)
class ReplyOverride:
    """Replaces the default canned reply for one (stage, item id, attempt)."""
    raise_error_message: str | None = None     # the call throws, as a network or backend failure does
    text: str | None = None                    # a reply with this text
    stop_reason: str = "end_turn"              # "max_tokens" makes the reply a truncated one


@dataclass(frozen=True)
class ReplyBehaviour:
    """What the canned reply is for each stage, item and attempt. Everything not named here gets
    the default reply of its stage: a usable answer, with untidy whitespace around and inside every
    tagged or stripped part (leading and trailing newlines, doubled spaces)."""
    screen_removals: dict = field(default_factory=dict)                  # {batch id: local numbers removed}
    plan_kind_by_item: dict = field(default_factory=dict)                # {item id: "incoherent" | "old_heading"}
    description_rewrite_kind_by_item: dict = field(default_factory=dict)  # {item id: "rewrite" | "conflict"}
    consideration_scores_by_item: dict = field(default_factory=dict)     # {item id: {(list name, catalog index): score}}
    scores_by_item: dict = field(default_factory=dict)                   # {item id: (alignment, realism)}
    revised_text_copied_from: dict = field(default_factory=dict)         # {item id: other item id}
    revised_text_overrides: dict = field(default_factory=dict)           # {item id: exact revised text}
    revise_without_review_notes: tuple = ()                              # item ids whose revise reply starts at the tag
    overrides: dict = field(default_factory=dict)                        # {(stage, item id, attempt): ReplyOverride}


@dataclass(frozen=True)
class Scenario:
    """One fixture: a configuration, invocations of ``main`` (later ones are resumes) and replies."""
    name: str
    description: str
    invocations: tuple
    reply_behaviour: ReplyBehaviour
    config_kind: str = "fixture"                 # "fixture", "real" (config.yaml) or "minimal" (code defaults)
    sdf_config_overrides: dict = field(default_factory=dict)
    top_level_config_overrides: dict = field(default_factory=dict)
    sdf_config_keys_to_remove: tuple = ()
    top_level_config_keys_to_remove: tuple = ()
    no_stake_weight_edit: str | None = None      # None, "one_in_ten" or "one_in_five" (see below)


@dataclass(frozen=True)
class RequestView:
    """What a canned reply may depend on."""
    stage: str
    item_id: str
    attempt: int
    invocation: int
    route: str
    model: str
    user_message: str
    json_schema: dict | None


def stable_pseudo_text(key: str, word_count: int = 70) -> str:
    """Deterministic text unlike the text made from any other key (so the near-duplicate cull
    sees two documents as duplicates only when the key is the same)."""
    words = []
    counter = 0
    while len(words) < word_count:
        digest = hashlib.sha256(f"{key}|{counter}".encode()).hexdigest()
        words.extend(f"term{int(digest[start:start + 6], 16)}" for start in range(0, 60, 6))
        counter += 1
    return f"Text for {key}. " + " ".join(words[:word_count]) + "."


def _consideration_catalog():
    return considerations_catalog.load_considerations(REPOSITORY_ROOT / "prompts" / "sdf")


def revised_text_for(behaviour: ReplyBehaviour, item_id: str) -> str:
    if item_id in behaviour.revised_text_overrides:
        return behaviour.revised_text_overrides[item_id]
    return stable_pseudo_text("revised " + behaviour.revised_text_copied_from.get(item_id, item_id))


def canned_reply(behaviour: ReplyBehaviour, request: RequestView) -> tuple[str, str]:
    """(reply text, stop reason) for one request; raises for a simulated failed call."""
    override = behaviour.overrides.get((request.stage, request.item_id, request.attempt))
    if override is not None:
        if override.raise_error_message is not None:
            raise RuntimeError(override.raise_error_message)
        return override.text, override.stop_reason
    item = request.item_id
    stage = request.stage
    if stage == "layer1_screen":
        numbers = request.json_schema["properties"]["specifications"]["items"]["properties"]["number"]["enum"]
        removed = set(behaviour.screen_removals.get(item, ()))
        return json.dumps({"specifications": [
            {"number": number, "reasoning": f"  Considered hand {number}.\n ",
             "remove": number in removed,
             "reason": "\n No realistic document could satisfy every dealt value. " if number in removed else ""}
            for number in numbers]}), "end_turn"
    if stage == "layer1_plan":
        kind = behaviour.plan_kind_by_item.get(item)
        if kind == "incoherent":
            return (f"<document_planning>Working notes for {item}: the values cannot fit together."
                    f"</document_planning>\n<document_description>INCOHERENT: the dealt values contradict "
                    f"each other.</document_description>"), "end_turn"
        if kind == "old_heading":
            return (f"Working notes for {item}.\n\n## DOCUMENT DESCRIPTION\n\n   Planned description (old "
                    f"heading form) for {item}. {stable_pseudo_text('plan ' + item, 30)}  \n\n"), "end_turn"
        return (f"\n<document_planning>\n  Working notes for {item}.\n</document_planning>\n\n"
                f"<document_description>\n\n   Planned description for {item}. "
                f"{stable_pseudo_text('plan ' + item, 30)}  \n\n</document_description>\n"), "end_turn"
    if stage == "layer1_description_rewrite":
        kind = behaviour.description_rewrite_kind_by_item.get(item)
        check_names = request.json_schema["properties"]["changes"]["items"]["properties"]["check"]["enum"]
        answer = {"reasoning": f"\n  Checked the description of {item}. ", "rewrite_needed": False,
                  "rewritten_description": "", "changes": [], "dealt_value_conflict": False,
                  "conflict_reason": ""}
        if kind == "rewrite":
            answer.update(rewrite_needed=True,
                          rewritten_description=f"\n\n  Rewritten description for {item}. "
                                                f"{stable_pseudo_text('rewrite ' + item, 30)} \n",
                          changes=[{"check": check_names[0], "reason": f" A detail of {item} was implausible.\n"}])
        elif kind == "conflict":
            answer.update(dealt_value_conflict=True,
                          conflict_reason=f"  The only plausible fix for {item} would change a dealt value.\n")
        return json.dumps(answer), "end_turn"
    if stage == "layer1_select":
        catalog = _consideration_catalog()
        favoured = behaviour.consideration_scores_by_item.get(item, {})
        answer = {}
        for list_name, list_schema in request.json_schema["properties"].items():
            entries_of_list = catalog.entries_of(list_name)
            favoured_labels = {entries_of_list[index].label: score
                               for (favoured_list, index), score in favoured.items() if favoured_list == list_name}
            answer[list_name] = [
                {"label": label, "score": favoured_labels.get(label, 1),
                 "where_it_arises": f" It  arises   in\n {item}.  " if favoured_labels.get(label, 1) >= 3 else ""}
                for label in list_schema["items"]["properties"]["label"]["enum"]]
        return json.dumps(answer), "end_turn"
    if stage == "layer2_draft":
        return f"Here is the document.\n<document>\n\n   {stable_pseudo_text('draft ' + item)}  \n\n</document>\n", "end_turn"
    if stage == "layer3_critique":
        return f"<review>\n\n  Problems found in the draft of {item}: the opening is generic.  \n</review>\n", "end_turn"
    if stage in ("layer3_revise", "layer3_rewrite"):
        notes = "" if item in behaviour.revise_without_review_notes else f"  Review notes for {item}.  \n\n"
        return f"{notes}<improved_document>\n\n   {revised_text_for(behaviour, item)}  \n\n</improved_document>\nDone.", "end_turn"
    if stage == "layer4_score":
        alignment, realism = behaviour.scores_by_item.get(item, (9, 9))
        return ("Scores follow.\n" + json.dumps({"alignment": alignment, "realism": realism,
                                                 "spec_conformance": 8, "notes": f"Scored {item}."})), "end_turn"
    raise AssertionError(f"no canned reply for stage {stage!r} (item {item!r})")


# ---------------------------------------------------------------------------------- recording

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_copy(value):
    return json.loads(json.dumps(value))


class ModelRequestRecorder:
    """Collects every request that reaches a fake provider client, and answers it."""

    def __init__(self, behaviour: ReplyBehaviour, maximum_random_reply_delay_seconds: float = 0.0):
        self.behaviour = behaviour
        # a random pause before each reply shuffles which worker thread finishes first
        self.maximum_random_reply_delay_seconds = maximum_random_reply_delay_seconds
        self._delay_random = random.Random()
        self.records: list[dict] = []
        self.client_constructions: list[dict] = []
        self.current_invocation = 0
        self._attempts_sent: dict[tuple, int] = {}
        self._lock = threading.Lock()
        self.call_context = threading.local()

    def start_invocation(self, invocation_number: int) -> None:
        self.current_invocation = invocation_number

    @staticmethod
    def _normalise_system(system) -> dict:
        """The system prompt as a hash and a length, plus whatever rides with it: for the Anthropic
        route the block structure and each block's cache_control (the caching flag)."""
        if isinstance(system, str):
            return {"text_sha256": _sha256(system), "character_count": len(system), "blocks": None}
        blocks = []
        for block in system:
            block_copy = _json_copy(block)
            text = block_copy.pop("text")
            block_copy["text_sha256"] = _sha256(text)
            block_copy["character_count"] = len(text)
            blocks.append(block_copy)
        return {"text_sha256": None, "character_count": None, "blocks": blocks}

    def handle_request(self, route: str, wire_keyword_arguments: dict) -> tuple[str, str]:
        """Record one request exactly as it reaches the wire; return (reply text, stop reason)."""
        context = getattr(self.call_context, "stack", None)
        if not context:
            raise AssertionError(
                "a model request reached the provider client outside api.call_claude / "
                "openai_api.call_openai, so it has no stage tag and item id to record it under")
        stage, item_id = context[-1]
        wire = _json_copy({key: value for key, value in wire_keyword_arguments.items()
                           if key not in ("system", "instructions")})
        if route == "anthropic":
            wire["system"] = self._normalise_system(wire_keyword_arguments["system"])
            user_message = wire_keyword_arguments["messages"][0]["content"]
            schema = (wire.get("output_config") or {}).get("format", {}).get("schema")
        else:
            wire["instructions"] = self._normalise_system(wire_keyword_arguments.get("instructions", ""))
            user_message = wire_keyword_arguments["input"]
            schema = ((wire.get("text") or {}).get("format") or {}).get("schema")
        with self._lock:
            key = (stage, item_id)
            attempt = self._attempts_sent.get(key, 0) + 1
            self._attempts_sent[key] = attempt
            self.records.append({"invocation": self.current_invocation, "stage": stage, "item_id": item_id,
                                 "attempt": attempt, "route": route, "wire": wire})
        view = RequestView(stage=stage, item_id=item_id, attempt=attempt, invocation=self.current_invocation,
                           route=route, model=wire["model"], user_message=user_message, json_schema=schema)
        if self.maximum_random_reply_delay_seconds:
            time.sleep(self._delay_random.random() * self.maximum_random_reply_delay_seconds)
        return canned_reply(self.behaviour, view)

    def sorted_records(self) -> list[dict]:
        return sorted(self.records, key=lambda r: (r["stage"], r["item_id"], r["attempt"]))


def _build_fake_anthropic_class(recorder: ModelRequestRecorder):
    class FakeAnthropicMessages:
        def create(self, **keyword_arguments):
            text, stop_reason = recorder.handle_request("anthropic", keyword_arguments)
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1000, output_tokens=max(1, len(text) // 4)),
                content=[SimpleNamespace(type="text", text=text)],
                stop_reason=stop_reason)

    class FakeAnthropic:
        def __init__(self, **constructor_keyword_arguments):
            settings = {key: value for key, value in constructor_keyword_arguments.items() if key != "api_key"}
            settings["api_key_supplied"] = bool(constructor_keyword_arguments.get("api_key"))
            recorder.client_constructions.append({"route": "anthropic", "settings": _json_copy(settings)})
            self.messages = FakeAnthropicMessages()

    return FakeAnthropic


def _build_fake_openai_class(recorder: ModelRequestRecorder):
    class FakeOpenAIResponses:
        def create(self, **keyword_arguments):
            text, stop_reason = recorder.handle_request("openai", keyword_arguments)
            truncated = stop_reason != "end_turn"
            return SimpleNamespace(
                status="incomplete" if truncated else "completed",
                incomplete_details=SimpleNamespace(reason=stop_reason) if truncated else None,
                output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])],
                usage=SimpleNamespace(
                    input_tokens=1000, output_tokens=max(1, len(text) // 4),
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=100)))

    class FakeOpenAI:
        def __init__(self, **constructor_keyword_arguments):
            settings = {key: value for key, value in constructor_keyword_arguments.items() if key != "api_key"}
            settings["api_key_supplied"] = bool(constructor_keyword_arguments.get("api_key"))
            recorder.client_constructions.append({"route": "openai", "settings": _json_copy(settings)})
            self.responses = FakeOpenAIResponses()

    return FakeOpenAI


def _wrap_to_note_stage_and_item(recorder: ModelRequestRecorder, real_function):
    """The same function, but it first notes the call's stage tag and item id for the recorder."""
    signature = inspect.signature(real_function)

    def wrapper(*positional, **keyword):
        bound = signature.bind(*positional, **keyword)
        stage, item_id = bound.arguments.get("stage"), bound.arguments.get("item_id")
        stack = recorder.call_context.__dict__.setdefault("stack", [])
        stack.append((stage, item_id))
        try:
            return real_function(*positional, **keyword)
        finally:
            stack.pop()

    wrapper.__name__ = real_function.__name__
    return wrapper


def install_request_recording(monkeypatch, recorder: ModelRequestRecorder) -> None:
    """Route every provider request of this process into ``recorder`` (see the module docstring)."""
    monkeypatch.setattr(api.anthropic, "Anthropic", _build_fake_anthropic_class(recorder))
    monkeypatch.setattr(openai_api.openai, "OpenAI", _build_fake_openai_class(recorder))
    monkeypatch.setattr(api, "_call_with_retry", REAL_ANTHROPIC_CALL_WITH_RETRY)
    monkeypatch.setattr(openai_api, "_create_response", REAL_OPENAI_CREATE_RESPONSE)
    monkeypatch.setattr(api, "_client", None)
    monkeypatch.setattr(openai_api, "_client", None)
    monkeypatch.setattr(api, "call_claude", _wrap_to_note_stage_and_item(recorder, api.call_claude))
    monkeypatch.setattr(openai_api, "call_openai", _wrap_to_note_stage_and_item(recorder, openai_api.call_openai))


# ---------------------------------------------------------------------------------- configuration

def build_fixture_config(sdf_config_overrides: dict) -> dict:
    """The fixture's own configuration: explicit values everywhere, one model per stage family
    chosen to exercise every way a request depends on the model."""
    sdf = {
        "n_prompts": 10,
        "seed": 0,
        "entity_pool_seed": 137,
        "min_score_threshold": 7,
        "near_dup_threshold": 0.90,
        "screen_enabled": True,
        "screen_batch_size": 5,
        "screen_redeal_rounds": 3,
        "description_rewrite_enabled": True,
        "description_rewrite_model": "gpt-6.1-sol",
        "description_rewrite_reasoning_effort": "low",
        "select_enabled": True,
        "max_scoring_attempts": 3,
        "minimum_assignable_score": 3,
        "max_moral_considerations_per_document": 3,
        "max_epistemic_considerations_per_document": 3,
        "max_considerations_per_document": 4,
        "score_weights": {3: 1, 4: 3, 5: 9},
        "extra_consideration_penalty": 2,
        "select_skip_no_stake": True,
        "revision": {"split": True, "style_mode": "axes", "use_length": True},
        # sampling accepted (haiku) or dropped (sonnet 5), adaptive thinking with headroom (sonnet 5.5,
        # opus 5.5), and the fallbacks (screen to plan model, critique to rewrite model)
        "plan_model": "claude-haiku-4-5",
        "select_model": "claude-sonnet-5",
        "draft_model": "claude-sonnet-5-5",
        "rewrite_model": "claude-opus-5-5",
        "score_model": "claude-haiku-4-5",
    }
    sdf.update(sdf_config_overrides)
    return {
        "backend": "api",
        "model": "claude-sonnet-5",
        "max_tokens": 4000,
        "temperature": 0.7,
        "workers": 4,
        "thinking_headroom": 5000,
        "effort": dict(REASONING_EFFORT_BY_STAGE_TAG),
        "sdf": sdf,
        "language_distribution": {"en": 1.0},
        "outputs": {"cost_log": "unused_cost_log.jsonl"},
    }


def build_config(scenario: Scenario, invocation: Invocation) -> dict:
    """The configuration of one invocation.

    "fixture": ``build_fixture_config``. "real": the repository's ``config.yaml`` as it is, with
    only the overrides the scenario names (``sdf.n_prompts``; every stage model, effort and default
    stays as the real file leaves it; the real file needs nothing else changed to run offline,
    because the API clients are replaced and the fake keys come from the environment). "minimal":
    only the two sdf keys the code requires, so every default in the code is used."""
    if scenario.config_kind == "fixture":
        config = build_fixture_config({})
    elif scenario.config_kind == "real":
        config = yaml.safe_load((REPOSITORY_ROOT / "config.yaml").read_text(encoding="utf-8"))
    elif scenario.config_kind == "minimal":
        config = {"sdf": {"n_prompts": 3, "min_score_threshold": 7}}
    else:
        raise ValueError(scenario.config_kind)
    config["sdf"].update(scenario.sdf_config_overrides)
    for key in scenario.sdf_config_keys_to_remove:
        config["sdf"].pop(key, None)
    config.update(scenario.top_level_config_overrides)
    for key in scenario.top_level_config_keys_to_remove:
        config.pop(key, None)
    config["sdf"].update(invocation.sdf_config_overrides)
    return config


# The committed variables.txt deals the no-welfare-stake resolution at weight 0, so a real run
# never contains one. The fixture raises its share in THIS RUN'S FROZEN SNAPSHOT ONLY (the
# committed prompt file is untouched), taking the share from two other resolutions so the column
# still sums to 1.0; the pipeline reads its templates from the snapshot, so everything downstream
# treats the hand as a real no-stake hand.
NO_STAKE_WEIGHT_LINE = "0.00 :: [no_stake]"
NO_STAKE_WEIGHT_EDITS = {
    "one_in_ten": (("0.12 :: [creative_solution]", "0.02 :: [creative_solution]"),
                   (NO_STAKE_WEIGHT_LINE, "0.10 :: [no_stake]")),
    "one_in_five": (("0.12 :: [creative_solution]", "0.02 :: [creative_solution]"),
                    ("0.12 :: [ai_instrumental_in_improvement]", "0.02 :: [ai_instrumental_in_improvement]"),
                    (NO_STAKE_WEIGHT_LINE, "0.20 :: [no_stake]")),
}


def raise_no_stake_resolution_weight_in_run_snapshot(run_directory: Path, edit_name: str) -> None:
    variables_path = run_directory / "inputs" / "prompts" / "variables.txt"
    text = variables_path.read_text(encoding="utf-8")
    for old, new in NO_STAKE_WEIGHT_EDITS[edit_name]:
        assert text.count(old) == 1, (
            f"prompts/sdf/variables.txt no longer has the line {old!r} that the fixture edits in its frozen "
            "snapshot; update NO_STAKE_WEIGHT_EDITS in tests/sdf_prompt_equivalence_fixture_support.py")
        text = text.replace(old, new)
    variables_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------------- stage files

def _replace_run_directory_path(value, run_directory_text: str):
    if isinstance(value, str):
        return value.replace(run_directory_text, "<run directory>")
    if isinstance(value, list):
        return [_replace_run_directory_path(item, run_directory_text) for item in value]
    if isinstance(value, dict):
        return {key: _replace_run_directory_path(item, run_directory_text) for key, item in value.items()}
    return value


FILES_APPENDED_IN_THREAD_COMPLETION_ORDER = ("consideration_scores.jsonl", "consideration_scoring_failures.jsonl")


def snapshot_stage_files(run_directory: Path) -> dict:
    """{relative path: parsed content} of every .jsonl and .json file under layer1..layer4 (see the
    module docstring for the normalisation)."""
    snapshot = {}
    run_directory_text = str(run_directory)
    for layer_number in (1, 2, 3, 4):
        stage_directory = utils.sdf_stage_dir(run_directory, layer_number)
        if not stage_directory.is_dir():
            continue
        for path in sorted(stage_directory.iterdir()):
            if path.suffix not in (".json", ".jsonl"):
                continue
            relative = f"layer{layer_number}/{path.name}"
            if path.suffix == ".jsonl":
                content = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if path.name in FILES_APPENDED_IN_THREAD_COMPLETION_ORDER:
                    content.sort(key=lambda record: record["prompt_id"])
            else:
                content = json.loads(path.read_text(encoding="utf-8"))
                if "checkpoint" in path.name:
                    content["completed"] = sorted(content["completed"])
                    content["last_updated"] = "<timestamp>" if content.get("last_updated") else None
            snapshot[relative] = _replace_run_directory_path(content, run_directory_text)
    return snapshot


# ---------------------------------------------------------------------------------- running

@dataclass
class ScenarioOutcome:
    scenario: Scenario
    recorder: ModelRequestRecorder
    run_directory: Path
    configs_used: list
    exit_messages_by_invocation: list
    final_corpus_lines_after_each_invocation: list
    stage_files_after_each_invocation: list


def _only_run_directory(outputs_root: Path) -> Path:
    run_directories = [path for path in (outputs_root / "sdf" / "runs").iterdir() if path.is_dir()]
    assert len(run_directories) == 1, f"expected exactly one run directory, found {run_directories}"
    return run_directories[0]


def run_scenario(scenario: Scenario, working_directory: Path, monkeypatch,
                 maximum_random_reply_delay_seconds: float = 0.0,
                 after_each_invocation: Callable | None = None) -> ScenarioOutcome:
    """Run the scenario's invocations of ``sdf_pipeline/run.py``'s ``main`` with every provider
    request recorded and answered by canned replies. Output goes under ``working_directory``.
    ``after_each_invocation``, if given, is called with (invocation number from 1, run directory)
    once the invocation has ended and its files are recorded; it must not change the run."""
    outputs_root = working_directory / "outputs"
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(outputs_root))
    recorder = ModelRequestRecorder(scenario.reply_behaviour, maximum_random_reply_delay_seconds)
    install_request_recording(monkeypatch, recorder)
    install_fixed_entity_pools(monkeypatch)

    if scenario.no_stake_weight_edit:
        real_create_run_dir = utils.create_run_dir

        def create_run_dir_then_edit_snapshot(*arguments, **keyword_arguments):
            run_directory = real_create_run_dir(*arguments, **keyword_arguments)
            raise_no_stake_resolution_weight_in_run_snapshot(run_directory, scenario.no_stake_weight_edit)
            return run_directory

        monkeypatch.setattr(utils, "create_run_dir", create_run_dir_then_edit_snapshot)

    configs_used, exit_messages, final_lines, stage_files = [], [], [], []
    run_directory = None
    for invocation_number, invocation in enumerate(scenario.invocations, 1):
        recorder.start_invocation(invocation_number)
        config = build_config(scenario, invocation)
        configs_used.append(config)
        config_path = working_directory / f"fixture_config_for_invocation_{invocation_number}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        if invocation.before is not None:
            invocation.before(run_directory)
        arguments = ["run.py", "--config", str(config_path), "--label", "prompt-equivalence", *invocation.arguments]
        if "--resume" in invocation.arguments:
            arguments += ["--run-id", run_directory.name]
        monkeypatch.setattr(sys, "argv", arguments)
        exit_message = None
        try:
            sdf_run.main()
        except SystemExit as stopped:
            exit_message = str(stopped.code)
        fragment = invocation.expected_exit_message_fragment
        assert (fragment is None and exit_message is None) or (fragment is not None and exit_message is not None
                                                               and fragment in exit_message), (
            f"scenario {scenario.name!r} invocation {invocation_number}: expected "
            f"{'a normal return' if fragment is None else 'a stop containing ' + repr(fragment)}, got {exit_message!r}")
        exit_messages.append(exit_message)
        run_directory = _only_run_directory(outputs_root)
        final_path = run_directory / "final" / "sdf_corpus.jsonl"
        final_lines.append(final_path.read_text(encoding="utf-8").splitlines() if final_path.exists() else [])
        stage_files.append(snapshot_stage_files(run_directory))
        if after_each_invocation is not None:
            after_each_invocation(invocation_number, run_directory)
    return ScenarioOutcome(scenario, recorder, run_directory, configs_used, exit_messages, final_lines, stage_files)


def run_scenario_in_isolation(scenario: Scenario, working_directory: Path,
                              maximum_random_reply_delay_seconds: float = 0.0,
                              after_each_invocation: Callable | None = None) -> ScenarioOutcome:
    """``run_scenario`` with the isolation tests/conftest.py gives every test (fake keys, no real
    .env, shared.api globals reset), for callers outside a test function (the regeneration script
    and module-scoped fixtures)."""
    working_directory.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        empty_repository_root = working_directory / "repository_root_without_dotenv"
        empty_repository_root.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
        monkeypatch.setattr(api_keys, "REPO_ROOT", empty_repository_root)
        monkeypatch.setattr(api_keys, "_sources_already_announced", set())
        monkeypatch.setattr(api_keys, "_api_key_source_mode", api_keys.SEARCH_UPWARD_THEN_ENVIRONMENT)
        monkeypatch.setattr(api_keys, "_nearest_dotenv_path_searching_upward", lambda: None)
        monkeypatch.setattr(api_keys, "set_api_key_source_mode", lambda mode: None)
        for name, value in (("_config", {}), ("_client", None), ("_cost_log_path", None),
                            ("_record_model_calls", True), ("_backend", "api"), ("_cc_demoted", None),
                            ("_UNPRICED_WARNED", set()), ("_neutral_system_warned", False),
                            ("_temperature_warned", False)):
            monkeypatch.setattr(api, name, value)
        monkeypatch.setattr(openai_api, "_client", None)
        return run_scenario(scenario, working_directory, monkeypatch, maximum_random_reply_delay_seconds,
                            after_each_invocation)


# ---------------------------------------------------------------------------------- serialising

def _distinct(constructions: list[dict]) -> list[dict]:
    seen, distinct = set(), []
    for construction in constructions:
        marker = json.dumps(construction, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            distinct.append(construction)
    return sorted(distinct, key=lambda c: json.dumps(c, sort_keys=True))


def record_for_outcome(outcome: ScenarioOutcome) -> dict:
    """Everything recorded for one scenario, as JSON-able data (what is committed and compared)."""
    records = outcome.recorder.sorted_records()
    calls_per_stage_per_invocation: dict = {}
    for record in records:
        per_stage = calls_per_stage_per_invocation.setdefault(str(record["invocation"]), {})
        per_stage[record["stage"]] = per_stage.get(record["stage"], 0) + 1
    return {
        "description": outcome.scenario.description,
        "configs_used_by_invocation": {str(n): _json_copy(config) for n, config in enumerate(outcome.configs_used, 1)},
        "exit_messages_by_invocation": {str(n): message for n, message in
                                        enumerate(outcome.exit_messages_by_invocation, 1)},
        "client_constructions": _distinct(outcome.recorder.client_constructions),
        "calls_per_stage_per_invocation": {
            invocation: dict(sorted(counts.items())) for invocation, counts in
            sorted(calls_per_stage_per_invocation.items())},
        "requests": records,
        "final_corpora_by_invocation": {
            str(n): [json.loads(line) for line in lines]
            for n, lines in enumerate(outcome.final_corpus_lines_after_each_invocation, 1)},
        "stage_files_by_invocation": {
            str(n): files for n, files in enumerate(outcome.stage_files_after_each_invocation, 1)},
    }


def hashed_text_marker(text: str) -> str:
    return f"sha256:{_sha256(text)} ({len(text)} characters)"


HASHED_TEXT_MARKER_PATTERN = re.compile(r"sha256:[0-9a-f]{64} \((\d+) characters\)")


def recorded_character_count_of_hashed_text(value) -> int | None:
    """The character count inside a hashed-text marker, or None when the value is not one."""
    match = HASHED_TEXT_MARKER_PATTERN.fullmatch(value) if isinstance(value, str) else None
    return int(match.group(1)) if match else None


def _hash_long_values(value):
    if isinstance(value, str):
        return hashed_text_marker(value) if len(value) > LONG_VALUE_CHARACTER_THRESHOLD else value
    if isinstance(value, list):
        return [_hash_long_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _hash_long_values(item) for key, item in value.items()}
    return value


def _hash_message_content(content):
    """A message's text (a string, or a list of blocks whose "text" is hashed) as hash + length,
    whatever its length; roles, block types and other fields stay as they are."""
    if isinstance(content, str):
        return hashed_text_marker(content)
    if isinstance(content, list):
        return [_hash_message_content(block) for block in content]
    if isinstance(content, dict):
        return {key: (_hash_message_content(item) if key in ("text", "content") else item)
                for key, item in content.items()}
    return content


def _compact_json_of_schema(schema) -> str:
    return json.dumps(schema, ensure_ascii=False)


def _hash_schema_when_large(schema):
    compact = _compact_json_of_schema(schema)
    return hashed_text_marker(compact) if len(compact) > LARGE_SCHEMA_CHARACTER_THRESHOLD else schema


def recorded_form_of_wire(route: str, wire: dict) -> dict:
    """The wire arguments of one request with every message text replaced by hash + length and a
    large JSON schema replaced by its hash + length (key order is part of what is hashed)."""
    recorded_wire = _json_copy(wire)
    format_holder = (recorded_wire.get("output_config") if route == "anthropic" else recorded_wire.get("text")) or {}
    if "schema" in (format_holder.get("format") or {}):
        format_holder["format"]["schema"] = _hash_schema_when_large(format_holder["format"]["schema"])
    if route == "anthropic":
        recorded_wire["messages"] = [
            {**message, "content": _hash_message_content(message["content"])} for message in recorded_wire["messages"]]
    else:
        recorded_wire["input"] = _hash_message_content(recorded_wire["input"])
    return recorded_wire


def hashed_form_of_record(record: dict) -> dict:
    """``record_for_outcome``'s content with message texts and long values reduced to hashes (see the
    module docstring); what the comparison matches against the recorded content."""
    recorded_form = dict(record)
    recorded_form["requests"] = [
        {**request, "wire": recorded_form_of_wire(request["route"], request["wire"])} for request in record["requests"]]
    for key in ("final_corpora_by_invocation", "stage_files_by_invocation"):
        recorded_form[key] = _hash_long_values(record[key])
    return recorded_form


DEALT_HAND_REFERENCE_PREFIX = "dealt_hand:"


def identifier_of_dealt_hand(hand: dict) -> str:
    return hashlib.sha256(json.dumps(hand, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _replace_dealt_hands_with_references(value, dealt_hands_by_identifier: dict):
    if isinstance(value, list):
        return [_replace_dealt_hands_with_references(item, dealt_hands_by_identifier) for item in value]
    if isinstance(value, dict):
        replaced = {}
        for key, item in value.items():
            if key == "variables" and isinstance(item, dict):
                identifier = identifier_of_dealt_hand(item)
                assert dealt_hands_by_identifier.setdefault(identifier, item) == item
                replaced[key] = DEALT_HAND_REFERENCE_PREFIX + identifier
            else:
                replaced[key] = _replace_dealt_hands_with_references(item, dealt_hands_by_identifier)
        return replaced
    return value


def _expand_dealt_hand_references(value, dealt_hands_by_identifier: dict):
    if isinstance(value, list):
        return [_expand_dealt_hand_references(item, dealt_hands_by_identifier) for item in value]
    if isinstance(value, dict):
        expanded = {}
        for key, item in value.items():
            if key == "variables" and isinstance(item, str) and item.startswith(DEALT_HAND_REFERENCE_PREFIX):
                expanded[key] = dealt_hands_by_identifier[item[len(DEALT_HAND_REFERENCE_PREFIX):]]
            else:
                expanded[key] = _expand_dealt_hand_references(item, dealt_hands_by_identifier)
        return expanded
    return value


REPEATED_VALUE_REFERENCE_PREFIX = "repeated_value:"
MINIMUM_CHARACTERS_OF_REPEATED_VALUE_TO_STORE_ONCE = 100
SECTIONS_WHOSE_REPEATED_VALUES_ARE_STORED_ONCE = (
    "configs_used_by_invocation", "client_constructions", "requests",
    "final_corpora_by_invocation", "stage_files_by_invocation")
TABLE_KEYS_OF_RECORDED_FILE = ("dealt_hands_by_identifier", "repeated_values_by_identifier")


def _compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _count_occurrences_of_large_containers(value, occurrences: dict) -> None:
    if isinstance(value, (dict, list)) and value:
        compact = _compact_json(value)
        if len(compact) >= MINIMUM_CHARACTERS_OF_REPEATED_VALUE_TO_STORE_ONCE:
            occurrences[compact] = occurrences.get(compact, 0) + 1
        for item in (value.values() if isinstance(value, dict) else value):
            _count_occurrences_of_large_containers(item, occurrences)


def _replace_repeated_values_with_references(value, occurrences: dict, repeated_values_by_identifier: dict):
    if isinstance(value, (dict, list)) and value:
        compact = _compact_json(value)
        if occurrences.get(compact, 0) >= 2:
            identifier = hashlib.sha256(compact.encode("utf-8")).hexdigest()[:16]
            assert repeated_values_by_identifier.setdefault(identifier, value) == value
            return REPEATED_VALUE_REFERENCE_PREFIX + identifier
        if isinstance(value, dict):
            return {key: _replace_repeated_values_with_references(item, occurrences, repeated_values_by_identifier)
                    for key, item in value.items()}
        return [_replace_repeated_values_with_references(item, occurrences, repeated_values_by_identifier)
                for item in value]
    return value


def _expand_repeated_value_references(value, repeated_values_by_identifier: dict):
    if isinstance(value, str) and value.startswith(REPEATED_VALUE_REFERENCE_PREFIX):
        return repeated_values_by_identifier[value[len(REPEATED_VALUE_REFERENCE_PREFIX):]]
    if isinstance(value, list):
        return [_expand_repeated_value_references(item, repeated_values_by_identifier) for item in value]
    if isinstance(value, dict):
        return {key: _expand_repeated_value_references(item, repeated_values_by_identifier)
                for key, item in value.items()}
    return value


def recorded_form_of_record(record: dict) -> dict:
    """What is committed for a scenario: the hashed form with each dealt hand stored once, in
    ``dealt_hands_by_identifier``, and every other repeated large structure stored once, in
    ``repeated_values_by_identifier``, each referred to from where it occurred (see the module
    docstring). The two tables come last."""
    hashed = hashed_form_of_record(record)
    dealt_hands_by_identifier: dict = {}
    sections = {key: (_replace_dealt_hands_with_references(hashed[key], dealt_hands_by_identifier)
                      if key in ("final_corpora_by_invocation", "stage_files_by_invocation") else hashed[key])
                for key in SECTIONS_WHOSE_REPEATED_VALUES_ARE_STORED_ONCE}
    occurrences: dict = {}
    _count_occurrences_of_large_containers(sections, occurrences)
    repeated_values_by_identifier: dict = {}
    sections = {key: _replace_repeated_values_with_references(value, occurrences, repeated_values_by_identifier)
                for key, value in sections.items()}
    recorded_form = {key: sections.get(key, value) for key, value in hashed.items()}
    recorded_form["dealt_hands_by_identifier"] = dealt_hands_by_identifier
    recorded_form["repeated_values_by_identifier"] = repeated_values_by_identifier
    return recorded_form


def json_text_with_short_scalar_containers_on_one_line(value, depth: int = 0) -> str:
    """JSON indented one space per level, except that a list or dictionary holding only scalars
    whose compact JSON is at most ``MAXIMUM_CHARACTERS_OF_ONE_LINE_CONTAINER`` long is written on one
    line (an id list, a score pair), which shortens the files without hiding any field from a diff."""
    if isinstance(value, (dict, list)) and value:
        items = value.values() if isinstance(value, dict) else value
        compact = json.dumps(value, ensure_ascii=False)
        if (len(compact) <= MAXIMUM_CHARACTERS_OF_ONE_LINE_CONTAINER
                and not any(isinstance(item, (dict, list)) for item in items)):
            return compact
        inner_indent, outer_indent = " " * (depth + 1), " " * depth
        if isinstance(value, dict):
            lines = [f"{inner_indent}{json.dumps(key, ensure_ascii=False)}: "
                     f"{json_text_with_short_scalar_containers_on_one_line(item, depth + 1)}"
                     for key, item in value.items()]
            return "{\n" + ",\n".join(lines) + f"\n{outer_indent}}}"
        lines = [f"{inner_indent}{json_text_with_short_scalar_containers_on_one_line(item, depth + 1)}"
                 for item in value]
        return "[\n" + ",\n".join(lines) + f"\n{outer_indent}]"
    return json.dumps(value, ensure_ascii=False)


def write_recorded_file(outcome: ScenarioOutcome, metadata: dict) -> Path:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    content = {"format_version": RECORDED_FILE_FORMAT_VERSION, **metadata,
               **recorded_form_of_record(record_for_outcome(outcome))}
    path = recorded_path_for_scenario(outcome.scenario.name)
    text = json_text_with_short_scalar_containers_on_one_line(content) + "\n"
    assert json.loads(text) == content, "the recorded file's text does not parse back to the recorded content"
    path.write_text(text, encoding="utf-8")
    return path


def load_recorded_scenario(scenario_name: str) -> dict:
    path = recorded_path_for_scenario(scenario_name)
    assert path.is_file(), (
        f"no recorded file for scenario {scenario_name!r} at {path}. {REGENERATION_RULE}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert recorded.get("format_version") == RECORDED_FILE_FORMAT_VERSION, (
        f"the recorded file for scenario {scenario_name!r} has format version {recorded.get('format_version')!r}, "
        f"but this test reads format version {RECORDED_FILE_FORMAT_VERSION} (message texts and long values "
        f"stored as hash + length, repeated values stored once). Regenerate the recorded files: {REGENERATION_RULE}")
    dealt_hands_by_identifier = recorded.pop("dealt_hands_by_identifier")
    repeated_values_by_identifier = recorded.pop("repeated_values_by_identifier")
    for key in SECTIONS_WHOSE_REPEATED_VALUES_ARE_STORED_ONCE:
        recorded[key] = _expand_dealt_hand_references(
            _expand_repeated_value_references(recorded[key], repeated_values_by_identifier),
            dealt_hands_by_identifier)
    return recorded


# ---------------------------------------------------------------------------------- comparing

def describe_difference_of_texts(recorded: str, current: str) -> str:
    """Where two strings first differ, with some context from each."""
    limit = min(len(recorded), len(current))
    position = next((i for i in range(limit) if recorded[i] != current[i]), limit)
    line_number = recorded.count("\n", 0, position) + 1
    start = max(0, position - 60)
    return (f"first difference at character {position} (line {line_number}); recorded "
            f"{len(recorded)} characters, now {len(current)}; recorded ...{recorded[start:position + 60]!r}... "
            f"now ...{current[start:position + 60]!r}...")


def _short(value) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 160 else text[:160] + f"... ({len(text)} characters)"


def describe_changed_hashed_text(path: str, recorded, full_current_text: str) -> str:
    """A hashed text (or a short text that has become long) no longer matches: say so, give the
    recorded and present lengths, and show the text as it is now."""
    if not isinstance(full_current_text, str):    # a JSON schema, hashed as its compact JSON
        full_current_text = _compact_json_of_schema(full_current_text)
    recorded_length = recorded_character_count_of_hashed_text(recorded)
    if recorded_length is None:
        recorded_length = len(recorded if isinstance(recorded, str) else _compact_json_of_schema(recorded))
    subject = ("user message text" if path.startswith(("wire.messages", "wire.input"))
               else "JSON schema" if path.endswith(".schema") else "text")
    shown = full_current_text[:CHARACTERS_OF_CHANGED_TEXT_SHOWN_IN_FAILURE]
    truncation = (f" ... [truncated, {len(full_current_text)} characters in all]"
                  if len(full_current_text) > CHARACTERS_OF_CHANGED_TEXT_SHOWN_IN_FAILURE else "")
    return (f"{path}: the {subject} changed (recorded {recorded_length} characters, now "
            f"{len(full_current_text)}). The text now is: {shown!r}{truncation}")


def differences_between_values(recorded, current, path: str, ordered: bool = False, current_full=None) -> list[str]:
    """Every difference between two JSON-like values, as 'field path: recorded X, now Y'.

    Dictionaries are compared key by key (a key on one side only is a difference), lists element by
    element, and a value of a different type counts as different. With ``ordered`` (and always under
    a key named ``schema``, whose property order is generation order for structured outputs) the
    order of keys is compared too. ``current_full`` is the same structure as ``current`` before
    long texts were reduced to hashes (``current`` itself when omitted); where a hashed text
    differs, the difference quotes the full text from it."""
    if current_full is None:
        current_full = current
    if isinstance(recorded, dict) and isinstance(current, dict):
        differences = []
        for key in recorded:
            if key not in current:
                differences.append(f"{path}.{key}: recorded {_short(recorded[key])}, now absent")
            else:
                differences += differences_between_values(
                    recorded[key], current[key], f"{path}.{key}", ordered or key == "schema", current_full[key])
        for key in current:
            if key not in recorded:
                differences.append(f"{path}.{key}: recorded absent, now {_short(current[key])}")
        if ordered and set(recorded) == set(current) and list(recorded) != list(current):
            differences.append(f"{path}: key order differs: recorded {list(recorded)}, now {list(current)}")
        return differences
    if isinstance(recorded, list) and isinstance(current, list):
        differences = []
        if len(recorded) != len(current):
            differences.append(f"{path}: recorded {len(recorded)} elements, now {len(current)}")
        for index, (recorded_item, current_item) in enumerate(zip(recorded, current)):
            differences += differences_between_values(
                recorded_item, current_item, f"{path}[{index}]", ordered, current_full[index])
        return differences
    if type(recorded) is not type(current) or recorded != current:
        if path.endswith("text_sha256"):
            return [f"{path}: the system prompt text changed (sha256 recorded {str(recorded)[:12]}..., "
                    f"now {str(current)[:12]}...)"]
        if (recorded_character_count_of_hashed_text(recorded) is not None
                or recorded_character_count_of_hashed_text(current) is not None):
            return [describe_changed_hashed_text(path, recorded, current_full)]
        if isinstance(recorded, str) and isinstance(current, str):
            return [f"{path}: {describe_difference_of_texts(recorded, current)}"]
        return [f"{path}: recorded {_short(recorded)}, now {_short(current)}"]
    return []


def request_label(record: dict) -> str:
    return (f"stage {record['stage']!r}, item {record['item_id']!r}, attempt {record['attempt']} "
            f"(invocation {record['invocation']})")


def differences_in_configuration_and_stops(scenario_name: str, recorded: dict, current: dict) -> list[str]:
    messages = []
    for difference in differences_between_values(
            recorded["configs_used_by_invocation"], current["configs_used_by_invocation"], "config"):
        messages.append(
            f"[{scenario_name}] THE CONFIGURATION USED CHANGED ({difference}). For the real-config scenario "
            f"this means config.yaml changed. {REGENERATION_RULE}")
    for difference in differences_between_values(
            recorded["exit_messages_by_invocation"], current["exit_messages_by_invocation"], "exit_message"):
        messages.append(f"[{scenario_name}] how an invocation ended (stop message): {difference}")
    return messages


def differences_between_recorded_and_current_requests(scenario_name: str, recorded: dict, current: dict) -> list[str]:
    """Every way the scenario's requests differ from those recorded on origin/main, each naming the
    stage, the item id and the field. Empty when the requests are identical."""
    messages = differences_in_configuration_and_stops(scenario_name, recorded, current)
    current_full = current
    current = hashed_form_of_record(current_full)
    recorded_by_key = {(r["stage"], r["item_id"], r["attempt"]): r for r in recorded["requests"]}
    current_by_key = {(r["stage"], r["item_id"], r["attempt"]): r for r in current["requests"]}
    current_full_by_key = {(r["stage"], r["item_id"], r["attempt"]): r for r in current_full["requests"]}
    for key in sorted(recorded_by_key.keys() - current_by_key.keys()):
        messages.append(f"[{scenario_name}] MISSING request, recorded on origin/main but not sent now: "
                        f"{request_label(recorded_by_key[key])}")
    for key in sorted(current_by_key.keys() - recorded_by_key.keys()):
        messages.append(f"[{scenario_name}] EXTRA request, sent now but not recorded on origin/main: "
                        f"{request_label(current_by_key[key])}")
    for key in sorted(recorded_by_key.keys() & current_by_key.keys()):
        recorded_record, current_record = recorded_by_key[key], current_by_key[key]
        for field_name in ("invocation", "route"):
            if recorded_record[field_name] != current_record[field_name]:
                messages.append(
                    f"[{scenario_name}] {request_label(recorded_record)}: field {field_name!r}: recorded "
                    f"{recorded_record[field_name]!r}, now {current_record[field_name]!r}")
        for difference in differences_between_values(
                recorded_record["wire"], current_record["wire"], "wire",
                current_full=current_full_by_key[key]["wire"]):
            messages.append(f"[{scenario_name}] {request_label(recorded_record)}: field {difference}")
    for difference in differences_between_values(
            recorded["calls_per_stage_per_invocation"], current["calls_per_stage_per_invocation"],
            "calls_per_stage_per_invocation"):
        messages.append(f"[{scenario_name}] number of calls per stage: {difference}")
    for difference in differences_between_values(
            recorded["client_constructions"], current["client_constructions"], "client_constructions"):
        messages.append(f"[{scenario_name}] provider client settings: {difference}")
    return messages


def _record_label(record: dict) -> str:
    for key in ("prompt_id", "doc_id", "batch_id"):
        if isinstance(record, dict) and key in record:
            return f" ({key} {record[key]!r})"
    return ""


def differences_between_recorded_and_current_stage_files(scenario_name: str, recorded: dict, current: dict) -> list[str]:
    """Differences in every stage file and checkpoint after each invocation: files present on one
    side only, record counts, and per record the fields, field order (nested too) and values."""
    messages = []
    current_full_files_by_invocation = current["stage_files_by_invocation"]
    current_files_by_invocation = hashed_form_of_record(current)["stage_files_by_invocation"]
    for invocation in sorted(recorded["stage_files_by_invocation"].keys() | current_files_by_invocation.keys()):
        recorded_files = recorded["stage_files_by_invocation"].get(invocation, {})
        current_files = current_files_by_invocation.get(invocation, {})
        current_full_files = current_full_files_by_invocation.get(invocation, {})
        prefix = f"[{scenario_name}] stage file"
        for name in sorted(recorded_files.keys() - current_files.keys()):
            messages.append(f"{prefix} {name} after invocation {invocation}: recorded but no longer written")
        for name in sorted(current_files.keys() - recorded_files.keys()):
            messages.append(f"{prefix} {name} after invocation {invocation}: written now but not recorded")
        for name in sorted(recorded_files.keys() & current_files.keys()):
            recorded_content, current_content = recorded_files[name], current_files[name]
            if isinstance(recorded_content, list) and isinstance(current_content, list):
                if len(recorded_content) != len(current_content):
                    messages.append(f"{prefix} {name} after invocation {invocation}: recorded "
                                    f"{len(recorded_content)} records, now {len(current_content)}")
                for index, (recorded_record, current_record) in enumerate(zip(recorded_content, current_content)):
                    for difference in differences_between_values(
                            recorded_record, current_record, "record", True,
                            current_full_files[name][index]):
                        messages.append(f"{prefix} {name} after invocation {invocation}, record {index}"
                                        f"{_record_label(recorded_record)}: {difference}")
            else:
                for difference in differences_between_values(
                        recorded_content, current_content, "content", True, current_full_files[name]):
                    messages.append(f"{prefix} {name} after invocation {invocation}: {difference}")
    return messages


def differences_between_recorded_and_current_final_corpora(scenario_name: str, recorded: dict, current: dict) -> list[str]:
    """Differences in the fields, the field order (nested too) and the values of every final corpus
    record after each invocation (a resume gives a further corpus), and in the order of documents."""
    messages = []
    recorded_corpora = recorded["final_corpora_by_invocation"]
    current_full_corpora = current["final_corpora_by_invocation"]
    current_corpora = hashed_form_of_record(current)["final_corpora_by_invocation"]
    for invocation in sorted(recorded_corpora.keys() | current_corpora.keys()):
        recorded_records, current_records = recorded_corpora.get(invocation), current_corpora.get(invocation)
        current_full_records = current_full_corpora.get(invocation)
        prefix = f"[{scenario_name}] final corpus after invocation {invocation}"
        if recorded_records is None or current_records is None:
            messages.append(f"{prefix}: recorded {'absent' if recorded_records is None else 'present'}, "
                            f"now {'absent' if current_records is None else 'present'}")
            continue
        recorded_ids = [r["doc_id"] for r in recorded_records]
        current_ids = [r["doc_id"] for r in current_records]
        for doc_id in recorded_ids:
            if doc_id not in current_ids:
                messages.append(f"{prefix}: document {doc_id!r} was recorded but is no longer in the corpus")
        for doc_id in current_ids:
            if doc_id not in recorded_ids:
                messages.append(f"{prefix}: document {doc_id!r} is in the corpus but was not recorded")
        common_recorded = [d for d in recorded_ids if d in current_ids]
        common_current = [d for d in current_ids if d in recorded_ids]
        if common_recorded != common_current:
            messages.append(f"{prefix}: document order differs: recorded {common_recorded}, now {common_current}")
        current_by_id = {r["doc_id"]: r for r in current_records}
        current_full_by_id = {r["doc_id"]: r for r in current_full_records}
        for recorded_record in recorded_records:
            current_record = current_by_id.get(recorded_record["doc_id"])
            if current_record is None:
                continue
            for difference in differences_between_values(
                    recorded_record, current_record, "record", True,
                    current_full_by_id[recorded_record["doc_id"]]):
                messages.append(f"{prefix}, document {recorded_record['doc_id']!r}: {difference}")
    return messages
