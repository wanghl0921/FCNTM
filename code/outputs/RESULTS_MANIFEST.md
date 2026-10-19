# RESULTS MANIFEST

- generated: 2026-10-19T03:31:25+00:00
- python: 3.13.15 on Windows-2019Server-10.0.17763-SP0
- protocol: K=30, seeds=[11, 23, 37], topn=10, read-outs=['beta', 'ctfidf']

## corpus

| path | sha256 | bytes |
|---|---|---|
| data/agnews/train.csv | `18ae93aaf03f25ea` | 1558082 |
| data/agnews/test.csv | `ffdaf569a16f1564` | 1956623 |
| data/github_issues/train.csv | `88b7e7b7133e5101` | 15900136 |
| data/github_issues/test.csv | `92d7657e57d350a5` | 4909113 |
| data/civil_openalex/raw.csv | `0d849556730aee05` | 16706536 |
| data/civil_openalex/train_full.csv | `90519f3de27df394` | 15249679 |
| data/civil_openalex/test_full.csv | `d1afccffccb6c35b` | 5032143 |
| data/civil_openalex/train_abs60.csv | `4b49d8ea9855eadc` | 10621036 |
| data/civil_openalex/test_abs60.csv | `ac24971a86801d54` | 3516883 |
| data/civil_openalex/train_abs30.csv | `dd7d4f7b0fa2fdea` | 9400226 |
| data/civil_openalex/test_abs30.csv | `9808049e3216c68e` | 3109210 |
| civil_corpus/collect_openalex.py | `e6fe8472f3e29c36` | 12414 |
| civil_corpus/civil_openalex_cc.csv | `0d849556730aee05` | 16706536 |
| agnews_repo/data/ag_news_csv/train.csv | `76a0a2d2f92b2863` | 29470338 |
| agnews_repo/data/ag_news_csv/test.csv | `521465c2428ed7f0` | 1857427 |

## checkpoint

| path | sha256 | bytes |
|---|---|---|
| outputs/fcntm_github_final_details.json | `8ab50a8a1aea2d46` | 22074540 |
| outputs/fcntm_agnews_final_details.json | `759b91932fa303e8` | 21183392 |
| outputs/fcntm_civil_final.csv | `9697121173ddf994` | 25237 |
| outputs/fcntm_civil_final_details.json | `2525e005e18c5fd1` | 27135792 |
| outputs/github_missing_state.pt | `6a2244fe94755e3a` | 37473191 |
| outputs/agnews_missing_state.pt | `e8c49f397b42269f` | 25767590 |
| outputs/github_fielddrop_off.pt | `1d228e61d5a287f0` | 28106983 |
| outputs/agnews_fielddrop_off.pt | `bf6f88d7160970ed` | 25766950 |
| outputs/civil_missing_state.pt | `00711cfc23df6ae5` | 82232515 |
| outputs/rerun/fcntm_main_state_github.pt | `c6f4f58b9f55ac7d` | 65585769 |
| outputs/rerun/fcntm_main_state_agnews.pt | `c8e68fadbbfb931c` | 60130921 |
| outputs/rerun/fcntm_main_state_civil.pt | `d203634b24f9de22` | 95942169 |
| outputs/rerun/concat_state_github.pt | `3660e6d7a5c6aa39` | 27469239 |
| outputs/rerun/concat_state_agnews.pt | `9eb1d04e48ae96f1` | 25131255 |
| outputs/rerun/concat_state_civil.pt | `a706754efc7e7897` | 40481079 |
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
| outputs/rerun/fcntm_github_seed53plus_details.json | `4c9b5097b8dc0d0a` | 51464263 |
| outputs/rerun/fcntm_agnews_seed53plus_details.json | `d10914eace834afc` | 49435487 |
| outputs/rerun/fair_readout_github_newseeds.csv | `8e9993ca8a8cb66a` | 1346 |
| outputs/rerun/fair_readout_agnews_newseeds.csv | `c777d4ac63c844e3` | 1343 |
| outputs/rerun/occlusion_robustness_github_newseeds.csv | `4c6282b0cd9122dd` | 6570 |
| outputs/rerun/occlusion_robustness_agnews_newseeds.csv | `df14a0e7c6389f37` | 7544 |
| outputs/rerun/seed_extension_summary.csv | `6ac0b0289a43df8b` | 1187 |
| outputs/rerun/fcntm_civil_seed53plus.csv | `70333ad926920048` | 58842 |
| outputs/rerun/fcntm_civil_seed53plus_details.json | `5bb3b970526bf704` | 63300520 |
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
| outputs/civil_openalex_topics_meta.json | `887bef9e537b0aac` | 626 |
| outputs/rerun/civil_topic_cases_20261014.md | `c5826f9cb0501ab6` | 5004 |
| outputs/rerun/concat_fix_numbers_20261014.txt | `655e087f0c226d85` | 1513 |

## code

| path | sha256 | bytes |
|---|---|---|
| config.py | `c67be6f0a0e7c00a` | 2072 |
| metrics.py | `ae3a1334c19ee66a` | 8638 |
| public_data.py | `486cd83cced65f3e` | 9353 |
| public_data_github.py | `a196448511c394fa` | 11474 |
| models.py | `ff1514bbbc4f9d01` | 72874 |
| baselines.py | `255d72ea1a40b962` | 18423 |
| new_baselines.py | `b349935350c8f5e7` | 13914 |
| experiment.py | `d06fc42b4905ab92` | 16914 |
| benchmark_public.py | `c7806050a778c9cd` | 18336 |
| run_fcntm_train.py | `a0fb2ad094334d8f` | 27302 |
| run_public_missing_field.py | `8308b303e87a68aa` | 30425 |
| occlusion_robustness.py | `a299ec9031339405` | 22978 |
| fair_readout.py | `9e06a2dc4e626cb3` | 6224 |
| rescore_clean_test.py | `ae2ab2fe45e45ecf` | 4177 |
| dump_btm_readout_topics.py | `c678c8b79734876a` | 3059 |
| add_btm_doc_topic.py | `c3ad08a14015a3a1` | 3777 |
| recompute_irbo.py | `163d972d41a4cb5b` | 2925 |
| run_ablation_public.py | `787246790fcce60b` | 19147 |
| run_core_ablations.py | `eeb83e60dc74590d` | 7045 |
| embed_public.py | `33e789f7970eab28` | 10383 |
| public_data_civil.py | `3ed59de9702c37b5` | 13809 |
| make_result_tables.py | `fe071cc99e01e05b` | 7406 |
| make_topic_table.py | `ebc9611309c1f757` | 7785 |
| compare_occlusion.py | `d342cf037611b312` | 5557 |
| check_paper_numbers.py | `c7e9a47438a71ac5` | 54339 |
| result_identity.py | `f1f836011d161046` | 7198 |
| rerun_btm_nmf_boundary_fix.py | `1797059f16dd19ed` | 6733 |
| check_empty_summary_behavior.py | `a83a981a7be0c21f` | 2697 |
| check_btm_empty_biterm_counts.py | `7a92d654ea808a95` | 2210 |
| check_missing_decomposition.py | `9cd684cebd4d8d90` | 2087 |
| results_manifest.py | `b607901c29df754c` | 25343 |
| field_redundancy.py | `247d6bb2c10baed8` | 7977 |
| field_complementarity.py | `120a9bde57ad0128` | 6673 |
| summarize_ladder.py | `16774b7217d64cc2` | 4774 |
| add_civil_btm_occlusion.py | `de5832bdc86c94dd` | 3903 |
| occlusion_controls.py | `f47da3270224452a` | 6048 |
| bn_readout_sensitivity.py | `f9f8bc01ee5ff04d` | 4987 |
| baseline_fieldview.py | `e4fec356766abd75` | 9549 |
| extend_seeds_btm.py | `1cfa48fc826e911c` | 3587 |
| extend_seeds_fcntm_occlusion.py | `1c72a0d9ef34e229` | 2152 |
| extend_seeds_lda_occlusion.py | `f34fcfa1c89d28b2` | 1732 |
| summarize_seed_extension.py | `4e8c175ec34f736b` | 9272 |
| seed_identity.py | `abfc2c7dd4ffcfe8` | 6293 |
| verify_newseed_archives.py | `6c22782cb46296f3` | 7569 |
| repin_fieldview_archive.py | `dbe6c160dd66161a` | 5477 |
| finish_fieldview_regen.py | `b841375a728a028f` | 1654 |
| run_concat_control.py | `50e10b84c8f1ff5a` | 11478 |
| civil_title_archival.py | `45541e634bb57859` | 13588 |
| summarize_civil_extension.py | `2e5569caeb600f33` | 6002 |
| fetch_openalex_topics.py | `68bc2d4daae2cf96` | 3549 |
| make_civil_table.py | `21a5378aab687c5f` | 3938 |
| make_civil_topic_cases_20261014.py | `527618aec8b8fb5c` | 7302 |
| check_english_numbers.py | `489c48b91ebefa81` | 18324 |

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
| main.pdf | `f863005e27b3b2bd` | 492112 |
| supplementary.tex | `140d0bf516f98f1a` | 19920 |
| supplementary.pdf | `65d78cc9c1d9f0ca` | 220672 |
| main_en.tex | `0d714da42d8ce697` | 89027 |
| main_en.pdf | `888e2026600de929` | 504267 |
| supplementary_en.tex | `34233fc51f16f985` | 24761 |
| supplementary_en.pdf | `b2be373c6fc99013` | 235437 |
| main_asce.tex | `8eb588bf8f711b00` | 73416 |
| main_asce.pdf | `b758807316795c2a` | 530533 |
| main_asce.bib | `99d51dcfd7759cec` | 14818 |
| FC-NTM_manuscript.docx | `dc162be51a024abb` | 672438 |
| FC-NTM_manuscript_ASCE.docx | `96e7d8eedf43efd8` | 669772 |
| FC-NTM_manuscript_zh.docx | `649f1831862e27f5` | 508035 |
| ascelike-new.cls | `b9002dae251b5666` | 25540 |
| ascelike-new.bst | `018056e4c6245316` | 34611 |
| make_paper_figures.py | `c2aa8e8944e61e61` | 10235 |
| topic_table_github_seed11_paper.tex | `7b40eb4d80501bda` | 1343 |
| fig_main_comparison.pdf | `7021ad9a168d02a6` | 25957 |
| fig_occlusion.pdf | `7414f63a018c716f` | 34049 |
| fig_occlusion_cross_corpus.pdf | `27d12812ae7a2be2` | 44187 |
| fig_complementarity.pdf | `e536447760a853c4` | 40578 |
| fig_architecture_en.pdf | `b402e903a1cd38bb` | 84635 |
| fig_seed_paired_en.pdf | `63f6ca5366fa5eb4` | 27565 |
