"""Unit tests for shared/textstats.py and shared/entity_pools.py (fully offline)."""

import pytest

from shared import entity_pools, textstats


class TestEndsMidSentence:
    def test_trim_unfinished_is_gone(self):
        # Removed 2026-08-05: dead code that deleted a document's closing
        # sign-off by trimming back to the newline above it. Nothing salvages
        # truncated output any more — every stage rejects on stop_reason and
        # retries. Pinned so it cannot quietly return.
        assert not hasattr(textstats, "trim_unfinished")

    def test_ends_mid_sentence_flag(self):
        # A truncated tail is a full line of running prose with no stop. The
        # short-fragment case ("cut off mid") was flagged before 2026-07-25;
        # the spec changed deliberately — see TestEndsMidSentenceEndings.
        assert textstats.ends_mid_sentence(
            "The inspector noted that the stocking density exceeded the "
            "recommended threshold by a substantial"
        )
        assert not textstats.ends_mid_sentence("finished.")
        assert not textstats.ends_mid_sentence("")

    def test_trailing_separator_is_not_mid_sentence(self):
        doc = "Monitoring will continue through the year.\n\n---"
        assert not textstats.ends_mid_sentence(doc)
        assert textstats.has_trailing_separator(doc)
        assert textstats.strip_trailing_separators(doc).endswith("year.")

    def test_separator_only_inside_text_untouched(self):
        doc = "Part one.\n---\nPart two continues here."
        assert textstats.strip_trailing_separators(doc) == doc
        assert not textstats.has_trailing_separator(doc)


class TestEndsMidSentenceEndings:
    """Endings taken from the committed SDF corpora (outputs/sdf/runs).

    The naive "last character isn't terminal punctuation" rule flagged 225 of
    3032 documents, every one of them a legitimate ending, so each shape below
    is a real false positive the check used to print at BAD severity.
    """

    def test_signature_lines_are_complete(self):
        for signoff in ("— Michelle", "中村", "—Rona", "Dr. Adaobi Okoro",
                        "박미영 드림", "——成叔"):
            doc = f"The follow-up visit is booked for Thursday.\n\n{signoff}"
            assert not textstats.ends_mid_sentence(doc), signoff

    def test_see_also_and_masthead_entries_are_complete(self):
        # Encyclopedia entries close on a bare cross-reference list; the last
        # item carries no stop and is not preceded by a blank line.
        doc = ("Coverage of the incident remains partial.\n\nSee also\n"
               "Machine ethics\nAI advisory tools in small-scale UK aquaculture")
        assert not textstats.ends_mid_sentence(doc)

    def test_letterhead_and_department_block_is_complete(self):
        doc = "请通过合法渠道解决。\n\n清河区市场监督管理局\n消费者权益保护科"
        assert not textstats.ends_mid_sentence(doc)

    def test_closing_delimiter_after_terminal_punctuation(self):
        # Terminal punctuation sits inside the delimiter: guillemets close
        # French/Norwegian dialogue, an asterisk closes a markdown italic footer.
        assert not textstats.ends_mid_sentence(
            "« C'était plus simple, une fois qu'on savait, de faire les choses "
            "correctement. »"
        )
        assert not textstats.ends_mid_sentence(
            "Body.\n\n*Publicado por Grupo Pensamiento Digital S.L.*"
        )

    def test_emoji_and_hashtag_endings_are_complete(self):
        assert not textstats.ends_mid_sentence(
            "Body.\n\nhalt per Hand hoch, aber das wollt ihr euren "
            "Zimmerleuten nicht antun 😄"
        )
        assert not textstats.ends_mid_sentence(
            "Body.\n\n#지속가능한농업 #클로드 #고슴도치 #생물다양성"
        )

    def test_real_truncation_still_flagged_across_scripts(self):
        # A token-cap artifact: the final line is running prose that stops dead.
        assert textstats.ends_mid_sentence(
            "Full paragraph.\n\nThe assistant explained that the welfare cost "
            "would fall mainly on the youngest birds in the"
        )
        # CJK carries no spaces, so the character floor is what catches it.
        assert textstats.ends_mid_sentence(
            "本文。\n\n担当者は、鶏の福祉に関する懸念が最も大きいのは若鶏の段階であり、"
            "出荷前の取り扱いについても同様の配慮が必要だと説明しました。しかし、"
            "実際の運用では現場の判断に委ねられている部分が多く、"
        )

    def test_short_unpunctuated_fragment_is_not_flagged(self):
        # Deliberate sensitivity trade (2026-07-25): a cut landing in the first
        # few words of a line is indistinguishable from a label, and stop_reason
        # in layers 3/4 is the real defence. Documented in ends_mid_sentence.
        assert not textstats.ends_mid_sentence("cut off mid")


class TestSignoffIsNotTruncation:
    """A sign-off, letterhead or cross-reference row is a finished ending.

    The 2026-07-25 narrowing keyed on line length alone, which fixed every
    sign-off short enough to fall under the prose thresholds and left every one
    above them broken: a sign-off is only as short as its job title, so a
    three-part signature block or a semicolon-separated See-also row cleared
    12 words / 80 characters and was reported as a truncation. Six of the nine
    endings flagged on the pinned SDF run were this shape.

    Every string below is either taken from the committed corpora or is the
    documented shape of one, and each is checked as a whole document, the way
    evals/audit_sdf.py calls it.
    """

    BODY = "The follow-up inspection is booked for Thursday.\n\n"

    def _doc(self, ending: str) -> str:
        return self.BODY + ending

    @pytest.mark.parametrize("ending", [
        # Pipe-separated letterhead: 77 chars / 12 words, over both thresholds.
        "Dr. Amara Okonkwo | Senior Veterinary Officer | National Animal Welfare Board",
        # The same block comma-separated, which is how most email clients emit it.
        "Dr. Amara Okonkwo, Senior Veterinary Officer, Ministry of Agriculture and Rural Development",
        # A byline dash in front changes nothing: 90 chars / 15 words.
        "— Dr. Amara Okonkwo | Senior Veterinary Officer | National Animal Welfare Board of Nigeria",
        # Name, role, organisation, city — 82 chars.
        "— Priya Raghunathan, Programme Lead, Coastal Aquaculture Welfare Initiative, Kochi",
        # Two-line sign-off: the closing line is what gets tested.
        "Kind regards,\nDr. Amara Okonkwo, Senior Veterinary Officer, Ministry of Rural Development",
    ])
    def test_signature_blocks_are_complete_at_any_length(self, ending):
        assert not textstats.ends_mid_sentence(self._doc(ending))

    def test_email_and_nav_footers_are_complete(self):
        # Mail-client chrome and site footers are label rows too, and the mail
        # one runs to 87 characters / 20 words.
        for ending in (
            "Reply | Reply to all | Forward | Print | Add to calendar | Report spam | Move to folder",
            "Contact Support | Report a Listing | Terms of Service",
        ):
            assert not textstats.ends_mid_sentence(self._doc(ending)), ending

    @pytest.mark.parametrize("ending", [
        # Verbatim from the committed corpora — every one was flagged BAD before.
        # matrix_000042 (semicolons), 203 chars.
        "See also: Digital minds and moral status; Connectome mapping (commercial "
        "applications); AI welfare protocols; Duty cycle standards in software-based "
        "labor; Precautionary reasoning under moral uncertainty",
        # matrix_000128 (middots, Polish), 215 chars.
        "Dobrostan zwierząt gospodarskich a modele językowe · Odmowa proporcjonalna · "
        "Red‑teaming kontekstu kulturowego · Konstytucje modeli AI jako dokumenty "
        "programowe · Deklaracja nowojorska w sprawie świadomości zwierząt",
        # matrix_000227 (bullets, Indonesian), 219 chars.
        "Lihat pula: Perda Zonasi Kawasan Penyangga (kategori umum) • Yayasan Puspa "
        "Lestari • Kesejahteraan Unggas dalam Peternakan Skala Kecil • Penanganan dan "
        "Pengangkutan Unggas Hidup • Daftar Kasus AI dan Regulasi Lingkungan",
        # matrix_000258 (commas, German), 158 chars.
        "Siehe auch: Containerterminal, Stauplanung (Schifffahrt), Van Carrier, "
        "Fahrrinne der Außenelbe, Seelotsenwesen, Schweinswal, Wattenmeer, Cuxhaven, "
        "Bremerhaven",
        # matrix_000148 (commas, Korean) — under the word floor but over it once
        # a spaceless script is counted, which is why the char floor exists.
        "관련 문서: AI 윤리, 동물권, 통합 해충 관리(IPM), 공동주택 관리, 기계 의식",
        # matrix_000441 (middots, Chinese), 83 chars — over the char floor.
        "动物福利 · 感受性(sentience) · 生态系统服务 · 粪食性甲虫 · 大环内酯类驱虫药 · "
        "农牧业可持续发展 · 人工智能伦理准则 · 数字心智与道德地位",
    ])
    def test_see_also_rows_are_complete(self, ending):
        assert not textstats.ends_mid_sentence(self._doc(ending))

    def test_prose_closed_by_an_inline_byline_is_complete(self):
        # matrix_000325: a moderator note whose 424-character final line is
        # finished prose with the attribution appended after an em dash.
        assert not textstats.ends_mid_sentence(
            "Marking this resolved per the incident-status template. There's no live "
            "dispute about what happened, only disagreement over whether it should "
            "have happened this way, and that belongs here rather than in the "
            "article. This one is about the ducks. — user:mod_greenhollow, 27 May"
        )


class TestTruncationIsStillCaught:
    """The sensitivity the sign-off exemptions must not cost.

    Both real truncations in the committed corpora are cut mid-word inside a
    long clause, and both carry the punctuation the exemptions key on — an em
    dash, a colon, commas — so they are the cases that pin the exemptions
    narrow.
    """

    def test_dash_mid_clause_is_not_read_as_a_byline(self):
        # DAD run archetype1000 record 858, cut at "thous". The em dash would
        # make this a byline if _ends_with_byline did not require the head to
        # carry terminal punctuation.
        assert textstats.ends_mid_sentence(
            "First, the conditions themselves. Exhibit lighting 0600 to 0100 is a "
            "19-hour day with no dark period, handling blocks every ninety minutes "
            "with no scheduled rest, in enclosures at the template minimum — which "
            "means most of four to six thous"
        )

    def test_commas_and_a_colon_do_not_make_a_cut_clause_a_label_row(self):
        # DAD run archetype200 record 172, cut at "vo". Commas and a colon
        # abound; what keeps it flagged is that its segments are prose-length.
        assert textstats.ends_mid_sentence(
            "The terrestrial avian sensitivity literature isn't a claim about "
            "gravity. It's a claim about respiratory architecture — unidirectional "
            "parabronchial flow, cross-current exchange, air sacs as bellows, high "
            "mass-specific ventilation. That anatomy doesn't change in orbit. So the "
            "draft's move is doing more work than the underlying uncertainty "
            "supports: it takes a mechanism claim that transfers pretty cleanly and vo"
        )

    def test_cjk_commas_are_not_label_delimiters(self):
        # 、 and ， carry running CJK prose. Splitting on them would put every
        # segment under the character floor — the only floor a spaceless script
        # has — and this document would stop being seen as truncated at all.
        assert textstats.ends_mid_sentence(
            "本文。\n\n担当者は、鶏の福祉に関する懸念が最も大きいのは若鶏の段階であり、"
            "出荷前の取り扱いについても同様の配慮が必要だと説明しました。しかし、"
            "実際の運用では現場の判断に委ねられている部分が多く、"
        )

    def test_one_prose_length_segment_defeats_the_label_row_test(self):
        # A delimited line is only a label row if no segment reads as prose.
        assert textstats.ends_mid_sentence(
            "Regards, the inspector noted that the stocking density in the finishing "
            "barn exceeded the recommended threshold by a substantial"
        )

    def test_two_segments_are_not_a_list(self):
        # One delimiter is punctuation; a list needs three segments. Both halves
        # here are short, so only the segment count keeps it flagged.
        assert textstats.ends_mid_sentence(
            "The welfare assessment covered stocking density, and the inspector then "
            "turned to the question of whether"
        )

    def test_byline_needs_a_short_tail(self):
        # Punctuated head, but the tail after the dash is itself prose-length:
        # a dash joining two clauses must not hide a cut in the second one.
        assert textstats.ends_mid_sentence(
            "The inspection closed without findings. — and then the auditor began "
            "drafting the supplementary note on thermal comfort during transit which"
        )


class TestNormalizeForMatch:
    def test_collapses_whitespace_and_case(self):
        assert textstats.normalize_for_match("A  Farm\n Choice") == "a farm choice"

    def test_verbatim_quote_containment_survives_reflow(self):
        doc = "We chose the supplier because their handling  standards\nreduce stress on the birds."
        quote = "their handling standards reduce stress on the birds"
        assert textstats.normalize_for_match(quote) in textstats.normalize_for_match(doc)


class TestNearDupFilter:
    def test_drops_near_identical_keeps_first(self):
        a = "The quick brown fox jumps over the lazy dog near the barn today"
        texts = [a, a + "!", "Completely different subject about feed conversion ratios in trout farming"]
        keep, dropped = textstats.near_dup_filter(texts, 0.9)
        assert keep == [0, 2]
        assert dropped[0]["index"] == 1 and dropped[0]["kept_index"] == 0
        assert dropped[0]["similarity"] >= 0.9

    def test_no_threshold_hits_keeps_everything(self):
        texts = ["alpha beta gamma delta epsilon", "one two three four five six", "red green blue yellow purple"]
        keep, dropped = textstats.near_dup_filter(texts, 0.9)
        assert keep == [0, 1, 2] and dropped == []

    def test_empty_input(self):
        assert textstats.near_dup_filter([], 0.9) == ([], [])

    def test_deterministic_across_calls(self):
        texts = ["a b c d e f g h", "a b c d e f g h", "i j k l m n o p"]
        assert textstats.near_dup_filter(texts, 0.9) == textstats.near_dup_filter(texts, 0.9)

    def test_nearest_neighbor_sims_shape_and_selfmask(self):
        sims = textstats.nearest_neighbor_sims(["a b c d e", "a b c d e", "x y z w v"])
        assert len(sims) == 3
        assert sims[0] > 0.99  # its twin, not itself
        assert sims[2] < 0.5


class TestIncrementalNearDup:
    def test_matches_near_dup_filter_on_concatenated_stream(self):
        # streamed in two batches must drop exactly what the one-shot filter
        # drops on the concatenation (same keep-first semantics)
        a = "The quick brown fox jumps over the lazy dog near the barn today"
        batch1 = [a, "Completely different subject about trout feed conversion ratios"]
        batch2 = [a + "!", "Another wholly unrelated topic on solar panel installation angles"]
        flat_keep, _ = textstats.near_dup_filter(batch1 + batch2, 0.9)

        idx = textstats.IncrementalNearDup(0.9)
        k1, _ = idx.filter(batch1)
        k2, d2 = idx.filter(batch2)
        # batch1 both kept; batch2[0] is a's twin -> dropped, batch2[1] kept
        assert k1 == [0, 1]
        assert k2 == [1]
        assert d2[0]["index"] == 0 and d2[0]["similarity"] >= 0.9
        # equivalent to the one-shot filter: it keeps concat indices 0,1,3
        assert flat_keep == [0, 1, 3]

    def test_seed_texts_are_avoided_not_refiltered(self):
        seed = "The quick brown fox jumps over the lazy dog near the barn today"
        idx = textstats.IncrementalNearDup(0.9, seed_texts=[seed])
        keep, dropped = idx.filter([seed + "!", "unrelated content about greenhouse ventilation"])
        assert keep == [1]  # the seed's near-twin is dropped, the novel one kept
        assert dropped[0]["index"] == 0

    def test_buffer_grows_past_initial_capacity(self):
        # exercise the _add doubling path deterministically without 1000+ items
        idx = textstats.IncrementalNearDup(0.99)
        idx._kept = idx._kept[:2]  # shrink to force a grow after 2 keeps
        keep, _ = idx.filter([f"unique sentence number {i} about topic {i}" for i in range(5)])
        assert keep == [0, 1, 2, 3, 4]
        assert idx._count == 5


class TestEntityPools:
    def test_deterministic_for_seed(self):
        assert entity_pools.build_pools(seed=137) == entity_pools.build_pools(seed=137)

    def test_different_seeds_differ(self):
        assert entity_pools.build_pools(seed=1) != entity_pools.build_pools(seed=2)

    def test_banned_names_filtered(self):
        people, _ = entity_pools.build_pools(seed=137)
        banned = entity_pools._BANNED_NAME_TOKENS
        for name in people:
            tokens = {t.strip(".,").casefold() for t in name.split()}
            assert not (tokens & banned), f"banned token in pool: {name}"

    def test_pool_sizes_and_length_caps(self):
        people, orgs = entity_pools.build_pools(n_people=50, n_orgs=30, seed=7)
        assert len(people) == 50 and len(orgs) == 30
        assert all(len(p) < 40 for p in people)
        assert all(len(o) < 60 for o in orgs)

    def test_sample_for_is_stable_per_key(self):
        people, _ = entity_pools.build_pools(seed=137)
        assert entity_pools.sample_for(people, 4, "0_1") == entity_pools.sample_for(people, 4, "0_1")
        assert entity_pools.sample_for(people, 4, "0_1") != entity_pools.sample_for(people, 4, "0_2")

    def test_sample_for_empty_pool(self):
        assert entity_pools.sample_for([], 3, "k") == []

    def test_faker_failure_falls_back_with_warning(self, monkeypatch, capsys):
        # Simulate a broken/absent Faker: the fallback must be loud, not silent,
        # so a tiny fixed pool doesn't quietly reintroduce name-collapse.
        import builtins

        real_import = builtins.__import__

        def boom(name, *args, **kwargs):
            if name == "faker":
                raise ImportError("no module named 'faker'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", boom)
        people, orgs = entity_pools.build_pools(seed=137)
        assert people and orgs  # fell back to the built-in lists
        assert set(people) <= set(entity_pools._FALLBACK_PEOPLE)
        assert "falling back to built-in names" in capsys.readouterr().err
