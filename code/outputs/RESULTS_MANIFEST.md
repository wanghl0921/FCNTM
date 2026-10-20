# RESULTS MANIFEST

- generated: 2026-10-20T07:05:45+00:00
- python: 3.13.15 on Windows-2019Server-10.0.17763-SP0
- protocol: K=30, seeds=[11, 23, 37], topn=10, read-outs=['beta', 'ctfidf']

## corpus

| path | sha256 | bytes |
|---|---|---|
| data/agnews/train.csv | `18ae93aaf03f25ea` | 1558082 |
| data/agnews/test.csv | `ffdaf569a16f1564` | 1956623 |
| data/github_issues/train.csv | `88b7e7b7133e5101` | 15900136 |
| data/github_issues/test.csv | `665db67457a4dc6c` | 4952908 |
| data/civil_openalex/raw.csv | `0d849556730aee05` | 16706536 |
| data/civil_openalex/train_full.csv | `90519f3de27df394` | 15249679 |
| data/civil_openalex/test_full.csv | `d1afccffccb6c35b` | 5032143 |
| data/civil_openalex/train_abs60.csv | `4b49d8ea9855eadc` | 10621036 |
| data/civil_openalex/test_abs60.csv | `ac24971a86801d54` | 3516883 |
| data/civil_openalex/train_abs30.csv | `dd7d4f7b0fa2fdea` | 9400226 |
| data/civil_openalex/test_abs30.csv | `9808049e3216c68e` | 3109210 |
| civil_corpus/collect_openalex.py | `5ec128294c2a3fae` | 12707 |
| civil_corpus/civil_openalex_cc.csv | `0d849556730aee05` | 16706536 |
| agnews_repo/data/ag_news_csv/train.csv | `8d425ab2441f8d31` | 29590338 |
| agnews_repo/data/ag_news_csv/test.csv | `79443885edce9f5f` | 1865027 |

## checkpoint

| path | sha256 | bytes |
|---|---|---|
| outputs/fcntm_github_final_details.json | **MISSING** | |
| outputs/fcntm_agnews_final_details.json | **MISSING** | |
| outputs/fcntm_civil_final.csv | `9697121173ddf994` | 25237 |
| outputs/fcntm_civil_final_details.json | **MISSING** | |
| outputs/github_missing_state.pt | **MISSING** | |
| outputs/agnews_missing_state.pt | **MISSING** | |
| outputs/github_fielddrop_off.pt | **MISSING** | |
| outputs/agnews_fielddrop_off.pt | **MISSING** | |
| outputs/civil_missing_state.pt | **MISSING** | |
| outputs/rerun/fcntm_main_state_github.pt | **MISSING** | |
| outputs/rerun/fcntm_main_state_agnews.pt | **MISSING** | |
| outputs/rerun/fcntm_main_state_civil.pt | **MISSING** | |
| outputs/rerun/concat_state_github.pt | **MISSING** | |
| outputs/rerun/concat_state_agnews.pt | **MISSING** | |
| outputs/rerun/concat_state_civil.pt | **MISSING** | |
| outputs/rerun/btm_fit_github_seed11.npz | `8a6479757359cc5e` | 197667 |
| outputs/rerun/btm_fit_github_seed23.npz | `09cd4ec326cdcb3a` | 195092 |
| outputs/rerun/btm_fit_github_seed37.npz | `713259b1123e312e` | 194645 |
| outputs/rerun/btm_fit_github_seed53.npz | `04d572abab37650e` | 1593306 |
| outputs/rerun/btm_fit_github_seed71.npz | `012b33e1fe6885f4` | 1594265 |
| outputs/rerun/btm_fit_github_seed89.npz | `438bf25783295082` | 1594593 |
| outputs/rerun/btm_fit_github_seed101.npz | `df69cfbfecb16646` | 1592719 |
| outputs/rerun/btm_fit_github_seed113.npz | `72b9afda7e3bd082` | 1592192 |
| outputs/rerun/btm_fit_github_seed127.npz | `9563664832760763` | 1594776 |
| outputs/rerun/btm_fit_github_seed139.npz | `3574a15a27896aa1` | 1594975 |
| outputs/rerun/btm_fit_agnews_seed11.npz | `bccab45b9d081ec1` | 205829 |
| outputs/rerun/btm_fit_agnews_seed23.npz | `2906047532a7121d` | 204526 |
| outputs/rerun/btm_fit_agnews_seed37.npz | `5414f9bb7d3e790b` | 204983 |
| outputs/rerun/btm_fit_agnews_seed53.npz | `092d49ffded20af3` | 1599776 |
| outputs/rerun/btm_fit_agnews_seed71.npz | `075385b89496484d` | 1598880 |
| outputs/rerun/btm_fit_agnews_seed89.npz | `5018a4c24e464444` | 1598926 |
| outputs/rerun/btm_fit_agnews_seed101.npz | `bd637458973478a6` | 1599003 |
| outputs/rerun/btm_fit_agnews_seed113.npz | `6e8bf3f963d656b4` | 1599460 |
| outputs/rerun/btm_fit_agnews_seed127.npz | `2e0f149ab95fba37` | 1598296 |
| outputs/rerun/btm_fit_agnews_seed139.npz | `4c65f5759bb50a8b` | 1599602 |
| outputs/rerun/nmf_fit_github_seed11.npz | `fc3fcd04d5387276` | 438747 |
| outputs/rerun/nmf_fit_github_seed23.npz | `6b80f3dca3c022e2` | 456770 |
| outputs/rerun/nmf_fit_github_seed37.npz | `ab44bd49e93a6d61` | 451157 |
| outputs/rerun/nmf_fit_agnews_seed11.npz | `eb2e913ae9af930b` | 287071 |
| outputs/rerun/nmf_fit_agnews_seed23.npz | `6ba9122f369b71e5` | 279814 |
| outputs/rerun/nmf_fit_agnews_seed37.npz | `ccfff80e86e7790e` | 293815 |

## result

| path | sha256 | bytes |
|---|---|---|
| outputs/public_benchmark_runs_agnews.csv | `6b803dda32c23e63` | 94000 |
| outputs/public_benchmark_runs_githubtrain6000_test2000.csv | `aa9a7272e2e3573b` | 152288 |
| outputs/rerun_btm_agnews.csv | `abd255e42657ff96` | 11356 |
| outputs/rerun_btm_github.csv | `74967ecb2080146e` | 11142 |
| outputs/rerun_btm_agnews_readouts.csv | `d6d2bc9d7b86727f` | 22161 |
| outputs/rerun_btm_github_readouts.csv | `9dd28a72321358b5` | 21707 |
| outputs/rerun_etm_agnews_fixed.csv | `5deb642d70dbf7b0` | 11517 |
| outputs/rerun_etm_github_fixed.csv | `17bf90b49b9fee29` | 11329 |
| outputs/rerun_combinedtm_agnews_fixed.csv | `46ca948f92aeb0f0` | 11675 |
| outputs/rerun_combinedtm_github_fixed.csv | `596c3ede18e6601f` | 11624 |
| outputs/fcntm_agnews_final.csv | `d604f226061733fb` | 22494 |
| outputs/fcntm_github_final.csv | `19f055ba295c9c8f` | 22662 |
| outputs/codev_agnews_gate.csv | `b7381df77891fb37` | 22325 |
| outputs/codev_github_gate_bal.csv | `bc15a626f05973ea` | 22665 |
| outputs/fair_readout_agnews_3seed.csv | `b10644e23c9bab70` | 3118 |
| outputs/fair_readout_github_3seed.csv | `aee7c8bddf9d4cd9` | 3122 |
| outputs/occlusion_robustness_github_seed11.csv | `2ecb829c47779096` | 1597 |
| outputs/occlusion_robustness_github_seed23.csv | `63aa5b8370e679cc` | 1599 |
| outputs/occlusion_robustness_github_seed37.csv | `b168f2fb0b38341b` | 1600 |
| outputs/occlusion_robustness_agnews_seed11.csv | `3298ac85f1b666ae` | 1900 |
| outputs/occlusion_robustness_agnews_seed23.csv | `041accc3d89c4745` | 1879 |
| outputs/occlusion_robustness_agnews_seed37.csv | `f4d9ab3748670ddf` | 1891 |
| outputs/occlusion_comparison.csv | `0812a42ec05f3093` | 4425 |
| outputs/clean_test_scores.csv | `c3804a970017c5dc` | 11809 |
| outputs/ablation_public_github_nofusion.csv | `be46fe5dc9b1b618` | 1028 |
| outputs/ablation_public_github_fusedmean.csv | `e6f842fc42318980` | 1035 |
| outputs/ablation_public_github_fusedmean_w0.csv | `89e7593de630de20` | 1023 |
| outputs/ablation_public_github_gate_w0p25.csv | `3acaf55836bcc7e7` | 1034 |
| outputs/ablation_public_github_gate_w1p0.csv | `46f74535acfea075` | 1029 |
| outputs/ablation_public_github_core.csv | `f5ff493a22e42304` | 3690 |
| outputs/ablation_public_agnews_core.csv | `2144496c173dedcf` | 3710 |
| outputs/ablation_public_civil_core.csv | `f37b1a0df164eb43` | 3662 |
| outputs/public_missing_field_occlusion.csv | `31cae27133afd808` | 6445 |
| outputs/public_missing_field_fieldview.csv | `e1874a9f52d04441` | 2321 |
| outputs/public_missing_field_gate.csv | `f32c6fc1e612f87a` | 829 |
| outputs/public_missing_field_agnews_gate.csv | `96530e0b860aadb6` | 489 |
| outputs/field_redundancy.json | `9446d767777e1987` | 794 |
| outputs/field_complementarity.csv | `3d45ad840ce725fa` | 782 |
| outputs/field_complementarity.json | `e82a651ed684944b` | 1591 |
| outputs/irbo_summary.csv | `85206acb5889e1b5` | 2001 |
| outputs/topic_table_github_seed11.tex | `d5f88adf445d7b52` | 7794 |
| outputs/rerun/btm_empty_biterm_counts.csv | `982ac7efe99360e3` | 864 |
| outputs/rerun/empty_summary_behavior.csv | `c9e0ebfc1a3782f2` | 226 |
| outputs/rerun/btm_empty_fallback_counts.csv | `aaf6f231c73d3563` | 167 |
| outputs/occlusion_robustness_civil_seed11.csv | `433eda4affca819d` | 1603 |
| outputs/occlusion_robustness_civil_seed23.csv | `9b5bb162dd162e8e` | 1601 |
| outputs/occlusion_robustness_civil_seed37.csv | `0498d6a1b4c41544` | 1602 |
| outputs/public_missing_field_civil_occlusion.csv | `67cff3006dbe04fa` | 3356 |
| outputs/public_missing_field_civil_fieldview.csv | `d4812d53533563f5` | 1228 |
| outputs/ablation_civil_abs60.csv | `2502dc818026a4b7` | 3043 |
| outputs/ablation_civil_abs30.csv | `1e0304df75c7b787` | 3058 |
| outputs/rerun/ladder_summary.csv | `79f3d6230feb3881` | 766 |
| outputs/rerun/chunked_runs_github.csv | `5dc2bc54a2702b44` | 84267 |
| outputs/rerun/chunked_benchmark_github.csv | `66136ed71fbe9196` | 1144 |
| outputs/rerun/chunked_runs_agnews.csv | `ea6e0da83dafcc1c` | 25010 |
| outputs/rerun/chunked_benchmark_agnews.csv | `923a8b58786170a3` | 1150 |
| outputs/rerun/chunked_v2_runs_github.csv | `dd202650956d4ac4` | 23223 |
| outputs/rerun/chunked_v2_benchmark_github.csv | `53443d059ced9a59` | 828 |
| outputs/rerun/chunked_v2_runs_agnews.csv | `d135bf05fc6d3e45` | 23268 |
| outputs/rerun/chunked_v2_benchmark_agnews.csv | `18a3b107fa95947b` | 830 |
| outputs/occlusion_control_unigram_github.csv | `1210c75e32777603` | 955 |
| outputs/occlusion_control_unigram_agnews.csv | `146a65008af45439` | 953 |
| outputs/occlusion_control_unigram_civil.csv | `fd532c1433fad0fb` | 941 |
| outputs/occlusion_control_shuffled_github.csv | `97f8bed932fabbd8` | 1239 |
| outputs/occlusion_control_shuffled_agnews.csv | `c0b2d23e010fbffe` | 1377 |
| outputs/occlusion_control_shuffled_civil.csv | `1d680b044eda0bbc` | 1224 |
| outputs/occlusion_control_fielddrop_off_github.csv | `58c9189f15412427` | 1239 |
| outputs/occlusion_control_fielddrop_off_agnews.csv | `20bd699e4f7698b8` | 1394 |
| outputs/occlusion_control_fielddrop_off_civil.csv | `17a51d87773d6999` | 1224 |
| outputs/bn_readout_sensitivity.csv | `23f25ac1398946e8` | 1870 |
| outputs/baseline_fieldview.csv | `6d04efccfdf447f3` | 6816 |
| outputs/rerun/fcntm_github_seed53plus.csv | `7cdb76e23b236b27` | 53233 |
| outputs/rerun/fcntm_agnews_seed53plus.csv | `785b75c795e81a9f` | 52870 |
| outputs/rerun/fcntm_github_seed53plus_details.json | **MISSING** | |
| outputs/rerun/fcntm_agnews_seed53plus_details.json | **MISSING** | |
| outputs/rerun/fair_readout_github_newseeds.csv | `8e9993ca8a8cb66a` | 1346 |
| outputs/rerun/fair_readout_agnews_newseeds.csv | `c777d4ac63c844e3` | 1343 |
| outputs/rerun/occlusion_robustness_github_newseeds.csv | `4c6282b0cd9122dd` | 6570 |
| outputs/rerun/occlusion_robustness_agnews_newseeds.csv | `df14a0e7c6389f37` | 7544 |
| outputs/rerun/seed_extension_summary.csv | `6ac0b0289a43df8b` | 1187 |
| outputs/rerun/fcntm_civil_seed53plus.csv | `70333ad926920048` | 58842 |
| outputs/rerun/fcntm_civil_seed53plus_details.json | **MISSING** | |
| outputs/rerun/fair_readout_civil_newseeds.csv | `ba7e4a4c809c0f44` | 1783 |
| outputs/rerun/occlusion_robustness_civil_newseeds.csv | `0f0af9ea18f74a3b` | 6459 |
| outputs/rerun/civil_seed_extension_summary.csv | `65ab5b796bc8d4e8` | 498 |
| outputs/fair_readout_civil_3seed.csv | `1fe731f9814ac313` | 4553 |
| outputs/rerun_main_baselines_civil.csv | `ae573a4641eba529` | 86895 |
| outputs/rerun_main_baselines_civil_summary.csv | `638beb6aa6751f37` | 1146 |
| outputs/civil_main_table.csv | `2fe27a8034e606f7` | 747 |
| outputs/rerun/concat_control_github.csv | `43d682444b5cbb6a` | 819 |
| outputs/rerun/concat_control_agnews.csv | `e8c3daeaa697e2be` | 820 |
| outputs/rerun/concat_control_civil.csv | `fedf4e9df2cc34f5` | 826 |
| outputs/rerun/civil_title_archival.csv | `30af9ac0566f4e0f` | 1398 |
| outputs/civil_openalex_topics.csv | `4e2f075217e7b4bc` | 1105979 |
| outputs/civil_openalex_topics_meta.json | `5b46f495acdb3f62` | 637 |
| outputs/rerun/civil_topic_cases_20261014.md | `c5826f9cb0501ab6` | 5004 |
| outputs/rerun/concat_fix_numbers_20261014.txt | `655e087f0c226d85` | 1513 |

## code

| path | sha256 | bytes |
|---|---|---|
| config.py | `370666cf1413c841` | 2129 |
| metrics.py | `561790773ae8d193` | 8692 |
| public_data.py | `486cd83cced65f3e` | 9353 |
| public_data_github.py | `b835d7a1c5a5a28b` | 11786 |
| models.py | `ff1514bbbc4f9d01` | 72874 |
| baselines.py | `7c8f44b723f85910` | 18924 |
| new_baselines.py | `bbb610469fdf2152` | 14257 |
| experiment.py | `a86c071ea7dc18f3` | 17409 |
| benchmark_public.py | `04955964722e447f` | 18602 |
| run_fcntm_train.py | `a0fb2ad094334d8f` | 27302 |
| run_public_missing_field.py | `9287e696480ba0ca` | 31084 |
| occlusion_robustness.py | `484d41c52f0b4206` | 23561 |
| fair_readout.py | `37988bc14871d936` | 6381 |
| rescore_clean_test.py | `fad26651263f45ab` | 4293 |
| dump_btm_readout_topics.py | `0b463e53f8df8ff2` | 3144 |
| add_btm_doc_topic.py | `1e96137956b2490a` | 3870 |
| recompute_irbo.py | `0fad87aacf74b49a` | 3009 |
| run_ablation_public.py | `787246790fcce60b` | 19147 |
| run_core_ablations.py | `353b7f0f35e1402d` | 7214 |
| embed_public.py | `33e789f7970eab28` | 10383 |
| public_data_civil.py | `3ed59de9702c37b5` | 13809 |
| make_result_tables.py | `01f8bd90e1b56764` | 7588 |
| make_topic_table.py | `33938f1c065b262e` | 7967 |
| compare_occlusion.py | `aca5f8dc0b939835` | 5690 |
| check_paper_numbers.py | `be9ef973118549fc` | 55572 |
| result_identity.py | `df5a7a3190f634f0` | 7410 |
| rerun_btm_nmf_boundary_fix.py | `4f8422e7f28497da` | 6892 |
| check_empty_summary_behavior.py | `4ae8e0c66690a8bc` | 2764 |
| check_btm_empty_biterm_counts.py | `08a44450437acbb0` | 2270 |
| check_missing_decomposition.py | `6bb61ad2c2067d4c` | 2142 |
| results_manifest.py | `7caa3bbf139719cf` | 25979 |
| field_redundancy.py | `aa88965080cbbdc9` | 8174 |
| field_complementarity.py | `120a9bde57ad0128` | 6673 |
| summarize_ladder.py | `df4a9a0d40a5e7a2` | 4889 |
| add_civil_btm_occlusion.py | `0f467ecc9ed76b5b` | 4003 |
| occlusion_controls.py | `c95d765e07f77190` | 6206 |
| bn_readout_sensitivity.py | `81c1fc015b9261bf` | 5100 |
| baseline_fieldview.py | `128aaecc3115afec` | 9763 |
| extend_seeds_btm.py | `b61cb3535777be4e` | 3676 |
| extend_seeds_fcntm_occlusion.py | `fe4a7870679d05d2` | 2208 |
| extend_seeds_lda_occlusion.py | `9acfa4f3d5343994` | 1781 |
| summarize_seed_extension.py | `3496de0490bbeaed` | 9500 |
| seed_identity.py | `ce3599f5a3d21160` | 6459 |
| verify_newseed_archives.py | `6c22782cb46296f3` | 7569 |
| repin_fieldview_archive.py | `9963de5f9647c780` | 5611 |
| finish_fieldview_regen.py | `f65a0035730d4401` | 1706 |
| run_concat_control.py | `691bfe35e4bde070` | 11768 |
| civil_title_archival.py | `f08aaff77482251a` | 13895 |
| summarize_civil_extension.py | `2615e23cee07a6a9` | 6148 |
| fetch_openalex_topics.py | `38fecaa2bf8450ea` | 3651 |
| make_civil_table.py | `3cad620be8c6d192` | 4042 |
| make_civil_topic_cases_20261014.py | `5c83dbe5d14b6fb7` | 7485 |
| check_english_numbers.py | `132533f0cb94f76e` | 18781 |

## table / figure -> artifacts

| label | artifacts |
|---|---|
| fig:seedpaired | outputs/fcntm_civil_final.csv<br>outputs/rerun/fcntm_civil_seed53plus.csv<br>outputs/fair_readout_civil_3seed.csv<br>outputs/rerun/fair_readout_civil_newseeds.csv<br>outputs/fcntm_github_final.csv<br>outputs/rerun/fcntm_github_seed53plus.csv<br>outputs/fair_readout_github_3seed.csv<br>outputs/rerun/fair_readout_github_newseeds.csv<br>outputs/fcntm_agnews_final.csv<br>outputs/rerun/fcntm_agnews_seed53plus.csv<br>outputs/fair_readout_agnews_3seed.csv<br>outputs/rerun/fair_readout_agnews_newseeds.csv |
| tab:agnews | outputs/public_benchmark_runs_agnews.csv<br>outputs/rerun_btm_agnews.csv<br>outputs/rerun_etm_agnews_fixed.csv<br>outputs/rerun_combinedtm_agnews_fixed.csv<br>outputs/fcntm_agnews_final.csv |
| tab:github | outputs/public_benchmark_runs_githubtrain6000_test2000.csv<br>outputs/rerun_btm_github.csv<br>outputs/rerun_etm_github_fixed.csv<br>outputs/rerun_combinedtm_github_fixed.csv<br>outputs/fcntm_github_final.csv |
| tab:fair | outputs/fair_readout_github_3seed.csv<br>outputs/fair_readout_agnews_3seed.csv<br>outputs/fcntm_github_final.csv<br>outputs/fcntm_agnews_final.csv |
| fig:main | outputs/public_benchmark_runs_agnews.csv<br>outputs/rerun_btm_agnews.csv<br>outputs/rerun_etm_agnews_fixed.csv<br>outputs/rerun_combinedtm_agnews_fixed.csv<br>outputs/fcntm_agnews_final.csv<br>outputs/public_benchmark_runs_githubtrain6000_test2000.csv<br>outputs/rerun_btm_github.csv<br>outputs/rerun_etm_github_fixed.csv<br>outputs/rerun_combinedtm_github_fixed.csv<br>outputs/fcntm_github_final.csv |
| tab:occl-robust | outputs/occlusion_robustness_github_seed11.csv |
| fig:occlusion | outputs/occlusion_robustness_github_seed11.csv |
| tab:occl-cross | outputs/occlusion_comparison.csv<br>outputs/occlusion_robustness_github_seed11.csv<br>outputs/occlusion_robustness_github_seed23.csv<br>outputs/occlusion_robustness_github_seed37.csv<br>outputs/occlusion_robustness_agnews_seed11.csv<br>outputs/occlusion_robustness_agnews_seed23.csv<br>outputs/occlusion_robustness_agnews_seed37.csv<br>outputs/occlusion_robustness_civil_seed11.csv<br>outputs/occlusion_robustness_civil_seed23.csv<br>outputs/occlusion_robustness_civil_seed37.csv |
| fig:occl-cross | outputs/occlusion_robustness_github_seed11.csv<br>outputs/occlusion_robustness_github_seed23.csv<br>outputs/occlusion_robustness_github_seed37.csv<br>outputs/occlusion_robustness_agnews_seed11.csv<br>outputs/occlusion_robustness_agnews_seed23.csv<br>outputs/occlusion_robustness_agnews_seed37.csv<br>outputs/occlusion_robustness_civil_seed11.csv<br>outputs/occlusion_robustness_civil_seed23.csv<br>outputs/occlusion_robustness_civil_seed37.csv |
| tab:occlusion | outputs/public_missing_field_occlusion.csv<br>outputs/public_missing_field_civil_occlusion.csv<br>outputs/civil_missing_state.pt |
| tab:fieldview | outputs/public_missing_field_fieldview.csv<br>outputs/public_missing_field_civil_fieldview.csv |
| tab:fieldview-baselines | outputs/baseline_fieldview.csv<br>outputs/public_missing_field_fieldview.csv<br>outputs/public_missing_field_civil_fieldview.csv |
| tab:fusion | outputs/ablation_public_github_nofusion.csv<br>outputs/ablation_public_github_fusedmean.csv<br>outputs/ablation_public_github_fusedmean_w0.csv<br>outputs/fcntm_github_final.csv |
| tab:weight | outputs/ablation_public_github_nofusion.csv<br>outputs/ablation_public_github_gate_w0p25.csv<br>outputs/ablation_public_github_gate_w1p0.csv<br>outputs/fcntm_github_final.csv |
| tab:core-ablation | outputs/ablation_public_github_core.csv<br>outputs/ablation_public_agnews_core.csv<br>outputs/ablation_public_civil_core.csv<br>outputs/fcntm_github_final.csv<br>outputs/fcntm_agnews_final.csv<br>outputs/fcntm_civil_final.csv |
| tab:conditional | outputs/field_redundancy.json<br>outputs/field_complementarity.csv<br>outputs/fair_readout_github_3seed.csv<br>outputs/fair_readout_agnews_3seed.csv<br>outputs/fcntm_github_final.csv<br>outputs/fcntm_agnews_final.csv |
| fig:cond | outputs/field_complementarity.csv<br>outputs/field_redundancy.json<br>outputs/fair_readout_github_3seed.csv<br>outputs/fair_readout_agnews_3seed.csv<br>outputs/fcntm_github_final.csv<br>outputs/fcntm_agnews_final.csv |
| tab:topics | outputs/topic_table_github_seed11.tex<br>outputs/fcntm_github_final_details.json |
| tab:ladder | outputs/rerun/ladder_summary.csv<br>outputs/ablation_civil_abs60.csv<br>outputs/ablation_civil_abs30.csv<br>outputs/ablation_public_civil_core.csv<br>outputs/fcntm_civil_final.csv<br>data/civil_openalex/train_full.csv<br>data/civil_openalex/train_abs60.csv<br>data/civil_openalex/train_abs30.csv |
| tab:chunked | outputs/rerun/chunked_v2_benchmark_github.csv<br>outputs/rerun/chunked_v2_benchmark_agnews.csv<br>outputs/public_benchmark_runs_githubtrain6000_test2000.csv<br>outputs/rerun_btm_github.csv<br>outputs/rerun_etm_github_fixed.csv<br>outputs/rerun_combinedtm_github_fixed.csv<br>outputs/fcntm_github_final.csv<br>outputs/public_benchmark_runs_agnews.csv<br>outputs/rerun_btm_agnews.csv<br>outputs/rerun_etm_agnews_fixed.csv<br>outputs/rerun_combinedtm_agnews_fixed.csv<br>outputs/fcntm_agnews_final.csv |
| tab:occl-controls | outputs/occlusion_control_unigram_github.csv<br>outputs/occlusion_control_unigram_agnews.csv<br>outputs/occlusion_control_unigram_civil.csv<br>outputs/occlusion_control_shuffled_github.csv<br>outputs/occlusion_control_shuffled_agnews.csv<br>outputs/occlusion_control_shuffled_civil.csv<br>outputs/occlusion_comparison.csv |
| tab:fielddrop-control | outputs/occlusion_control_fielddrop_off_github.csv<br>outputs/occlusion_control_fielddrop_off_agnews.csv<br>outputs/occlusion_control_fielddrop_off_civil.csv<br>outputs/occlusion_comparison.csv |
| tab:beta-readout | outputs/bn_readout_sensitivity.csv |
| tab:seed-paired | outputs/rerun/seed_extension_summary.csv<br>outputs/rerun/fcntm_github_seed53plus.csv<br>outputs/rerun/fcntm_agnews_seed53plus.csv<br>outputs/rerun/fair_readout_github_newseeds.csv<br>outputs/rerun/fair_readout_agnews_newseeds.csv<br>outputs/rerun/occlusion_robustness_github_newseeds.csv<br>outputs/rerun/occlusion_robustness_agnews_newseeds.csv |
| tab:clean-test | outputs/clean_test_scores.csv |
| tab:civil | outputs/fcntm_civil_final.csv<br>outputs/fair_readout_civil_3seed.csv<br>outputs/rerun_main_baselines_civil_summary.csv |
| tab:concat | outputs/rerun/concat_control_github.csv<br>outputs/rerun/concat_control_agnews.csv<br>outputs/rerun/concat_control_civil.csv<br>outputs/fcntm_github_final.csv<br>outputs/fcntm_agnews_final.csv<br>outputs/fcntm_civil_final.csv |
| tab:archival | outputs/rerun/civil_title_archival.csv<br>outputs/civil_openalex_topics.csv |
| tab:civil-cases | outputs/rerun/civil_topic_cases_20261014.md<br>outputs/fcntm_civil_final_details.json<br>outputs/civil_missing_state.pt |

## paper files

| file | sha256 | bytes |
|---|---|---|
| main.tex | `9d7baf212735cc3c` | 104644 |
| supplementary.tex | `140d0bf516f98f1a` | 19920 |
| main_en.tex | `c7b40439c9666ff0` | 85173 |
| supplementary_en.tex | `24bee07a971a1c84` | 42022 |
| main_asce.tex | `50464f9940e1d4c7` | 54715 |
| main_asce.bib | `99d51dcfd7759cec` | 14818 |
| ascelike-new.cls | `b9002dae251b5666` | 25540 |
| ascelike-new.bst | `018056e4c6245316` | 34611 |
| topic_table_github_seed11_paper.tex | `7b40eb4d80501bda` | 1343 |
| fig_main_comparison.pdf | `7021ad9a168d02a6` | 25957 |
| fig_occlusion.pdf | `7414f63a018c716f` | 34049 |
| fig_occlusion_cross_corpus.pdf | `27d12812ae7a2be2` | 44187 |
| fig_complementarity.pdf | `e536447760a853c4` | 40578 |
| fig_architecture_en.pdf | `9c564885ede64c13` | 87801 |
| fig_seed_paired_en.pdf | `437a77cdb3eb69c9` | 27172 |
