# RESULTS MANIFEST

- generated: 2026-10-20T08:03:33+00:00
- python: 3.13.15 on Windows-2019Server-10.0.17763-SP0
- protocol: K=30, seeds=[11, 23, 37], topn=10, read-outs=['beta', 'ctfidf']

## corpus

| path | sha256 | bytes |
|---|---|---|
| data/agnews/train.csv | `864e08b07b474f0d` | 1552081 |
| data/agnews/test.csv | `341939e876adfc14` | 1949022 |
| data/github_issues/train.csv | `88b7e7b7133e5101` | 15900136 |
| data/github_issues/test.csv | `6a007216498b373b` | 4837764 |
| data/civil_openalex/raw.csv | `ef726378c71976a8` | 16694297 |
| data/civil_openalex/train_full.csv | `40c6f83b174746fa` | 15243678 |
| data/civil_openalex/test_full.csv | `045cf974265bcdd3` | 5030142 |
| data/civil_openalex/train_abs60.csv | `ffdc74bc564861a9` | 10615035 |
| data/civil_openalex/test_abs60.csv | `b5297c4c0ad5895f` | 3514882 |
| data/civil_openalex/train_abs30.csv | `cd61db0fbf5995f5` | 9394225 |
| data/civil_openalex/test_abs30.csv | `737dfd77be67c32e` | 3107209 |
| civil_corpus/collect_openalex.py | `e6fe8472f3e29c36` | 12414 |
| civil_corpus/civil_openalex_cc.csv | `ef726378c71976a8` | 16694297 |
| agnews_repo/data/ag_news_csv/train.csv | `76a0a2d2f92b2863` | 29470338 |
| agnews_repo/data/ag_news_csv/test.csv | `521465c2428ed7f0` | 1857427 |

## checkpoint

| path | sha256 | bytes |
|---|---|---|
| outputs/fcntm_github_final_details.json | **MISSING** | |
| outputs/fcntm_agnews_final_details.json | **MISSING** | |
| outputs/fcntm_civil_final.csv | `be1abd0666aaae93` | 25230 |
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
| outputs/public_benchmark_runs_agnews.csv | `cbfbf1b85a23d3ca` | 93972 |
| outputs/public_benchmark_runs_githubtrain6000_test2000.csv | `795b37fb0bb555e3` | 152260 |
| outputs/rerun_btm_agnews.csv | `1f219f04859aed34` | 11352 |
| outputs/rerun_btm_github.csv | `abf0c920f0dabb63` | 11138 |
| outputs/rerun_btm_agnews_readouts.csv | `64cb3841ae98046b` | 22154 |
| outputs/rerun_btm_github_readouts.csv | `c30b00fb35ae5e5f` | 21700 |
| outputs/rerun_etm_agnews_fixed.csv | `f9834049114185d9` | 11513 |
| outputs/rerun_etm_github_fixed.csv | `952832cde7ecc186` | 11325 |
| outputs/rerun_combinedtm_agnews_fixed.csv | `e8457ce097a59071` | 11671 |
| outputs/rerun_combinedtm_github_fixed.csv | `b5425a59c42f0bf2` | 11620 |
| outputs/fcntm_agnews_final.csv | `8080f1ce6f8918bb` | 22487 |
| outputs/fcntm_github_final.csv | `882308480d19aa1e` | 22655 |
| outputs/codev_agnews_gate.csv | `8cb37242a397ae8d` | 22318 |
| outputs/codev_github_gate_bal.csv | `a89c88b3839b8f8f` | 22658 |
| outputs/fair_readout_agnews_3seed.csv | `900ab36602724c73` | 3081 |
| outputs/fair_readout_github_3seed.csv | `80f8547ad0a8a3e5` | 3085 |
| outputs/occlusion_robustness_github_seed11.csv | `eaf6e62f4c3ba4a5` | 1566 |
| outputs/occlusion_robustness_github_seed23.csv | `b2bd500d4fabe477` | 1568 |
| outputs/occlusion_robustness_github_seed37.csv | `9d6bdc1e81b2345b` | 1569 |
| outputs/occlusion_robustness_agnews_seed11.csv | `381ff6a201d894b9` | 1869 |
| outputs/occlusion_robustness_agnews_seed23.csv | `36aa1fe0b14f11e0` | 1848 |
| outputs/occlusion_robustness_agnews_seed37.csv | `338d951359fe1f11` | 1860 |
| outputs/occlusion_comparison.csv | `6541c95888ecf4c5` | 4370 |
| outputs/clean_test_scores.csv | `922e8dc9b8a1ccaf` | 11712 |
| outputs/ablation_public_github_nofusion.csv | `7d5ea97b0e0f7d79` | 1021 |
| outputs/ablation_public_github_fusedmean.csv | `750ea3100a282d2b` | 1028 |
| outputs/ablation_public_github_fusedmean_w0.csv | `09a0410920b99f5b` | 1016 |
| outputs/ablation_public_github_gate_w0p25.csv | `2d7c8d82e493b0ac` | 1027 |
| outputs/ablation_public_github_gate_w1p0.csv | `42e5eac30d040b1a` | 1022 |
| outputs/ablation_public_github_core.csv | `07d8558ec27d5aa8` | 3665 |
| outputs/ablation_public_agnews_core.csv | `fa9fef164e08c9ff` | 3685 |
| outputs/ablation_public_civil_core.csv | `c1cfbd4563597693` | 3637 |
| outputs/public_missing_field_occlusion.csv | `f63e564faf8b2225` | 6432 |
| outputs/public_missing_field_fieldview.csv | `a93e948a3b4c90e6` | 2296 |
| outputs/public_missing_field_gate.csv | `c6c8915a0aaed621` | 822 |
| outputs/public_missing_field_agnews_gate.csv | `1db1e48fa174a0d6` | 485 |
| outputs/field_redundancy.json | `db9654cf4b82c9ce` | 771 |
| outputs/field_complementarity.csv | `10e498d775b3161f` | 778 |
| outputs/field_complementarity.json | `d2699161e5f36927` | 1542 |
| outputs/irbo_summary.csv | `d54a8a365e8e75ca` | 1976 |
| outputs/topic_table_github_seed11.tex | `36043352139d4020` | 7732 |
| outputs/rerun/btm_empty_biterm_counts.csv | `987caad013dbbebf` | 839 |
| outputs/rerun/empty_summary_behavior.csv | `2e5ef04a17abe77c` | 223 |
| outputs/rerun/btm_empty_fallback_counts.csv | `b055fb7749a583c2` | 160 |
| outputs/occlusion_robustness_civil_seed11.csv | `f84a4ed8da2e808d` | 1572 |
| outputs/occlusion_robustness_civil_seed23.csv | `fe3db548d099c535` | 1570 |
| outputs/occlusion_robustness_civil_seed37.csv | `cb81828994b99a2f` | 1571 |
| outputs/public_missing_field_civil_occlusion.csv | `e8e6625504a85263` | 3349 |
| outputs/public_missing_field_civil_fieldview.csv | `16c9ff9618f8ff19` | 1215 |
| outputs/ablation_civil_abs60.csv | `6ccb8c8c6b36d2b0` | 3024 |
| outputs/ablation_civil_abs30.csv | `75dc664145f3b221` | 3039 |
| outputs/rerun/ladder_summary.csv | `d9f83748c4179aa3` | 753 |
| outputs/rerun/chunked_runs_github.csv | `e301fd3fb7067413` | 84257 |
| outputs/rerun/chunked_benchmark_github.csv | `05e95c5218cb126d` | 1140 |
| outputs/rerun/chunked_runs_agnews.csv | `4ed66533af9da9b4` | 25000 |
| outputs/rerun/chunked_benchmark_agnews.csv | `d259e7add3ebffaf` | 1146 |
| outputs/rerun/chunked_v2_runs_github.csv | `e57c018892bf5233` | 23216 |
| outputs/rerun/chunked_v2_benchmark_github.csv | `769221291f581048` | 825 |
| outputs/rerun/chunked_v2_runs_agnews.csv | `99d8fb3fa7080ea9` | 23261 |
| outputs/rerun/chunked_v2_benchmark_agnews.csv | `ba593f7433399938` | 827 |
| outputs/occlusion_control_unigram_github.csv | `8931bfb4f328a0d4` | 939 |
| outputs/occlusion_control_unigram_agnews.csv | `2e39ed57339163b8` | 937 |
| outputs/occlusion_control_unigram_civil.csv | `740d71df4a914acf` | 925 |
| outputs/occlusion_control_shuffled_github.csv | `dbd3297f3b4e6d64` | 1223 |
| outputs/occlusion_control_shuffled_agnews.csv | `5e0ebb2929139f6f` | 1361 |
| outputs/occlusion_control_shuffled_civil.csv | `4f23eef620b5c4a6` | 1208 |
| outputs/occlusion_control_fielddrop_off_github.csv | `61ee39094b35e34d` | 1223 |
| outputs/occlusion_control_fielddrop_off_agnews.csv | `30beffc272d50a50` | 1378 |
| outputs/occlusion_control_fielddrop_off_civil.csv | `8178af466826839d` | 1208 |
| outputs/bn_readout_sensitivity.csv | `ac2a82b363e87f47` | 1860 |
| outputs/baseline_fieldview.csv | `3b1e538fbc111711` | 6743 |
| outputs/rerun/fcntm_github_seed53plus.csv | `41ca727f4e864b40` | 53218 |
| outputs/rerun/fcntm_agnews_seed53plus.csv | `6d818669afb37de1` | 52855 |
| outputs/rerun/fcntm_github_seed53plus_details.json | **MISSING** | |
| outputs/rerun/fcntm_agnews_seed53plus_details.json | **MISSING** | |
| outputs/rerun/fair_readout_github_newseeds.csv | `92d136c49514dcc9` | 1331 |
| outputs/rerun/fair_readout_agnews_newseeds.csv | `adacb23db1f4e9a7` | 1328 |
| outputs/rerun/occlusion_robustness_github_newseeds.csv | `f063c657c7ccf6e3` | 6464 |
| outputs/rerun/occlusion_robustness_agnews_newseeds.csv | `e735f4a88a04244d` | 7438 |
| outputs/rerun/seed_extension_summary.csv | `3aa6b32e6f92cd18` | 1178 |
| outputs/rerun/fcntm_civil_seed53plus.csv | `30f4b1dab8832da4` | 58827 |
| outputs/rerun/fcntm_civil_seed53plus_details.json | **MISSING** | |
| outputs/rerun/fair_readout_civil_newseeds.csv | `1dcae4981672dc46` | 1768 |
| outputs/rerun/occlusion_robustness_civil_newseeds.csv | `2b61d78f075c250b` | 6353 |
| outputs/rerun/civil_seed_extension_summary.csv | `fdf5b72d24a92668` | 494 |
| outputs/fair_readout_civil_3seed.csv | `2a5053e22f593aff` | 4516 |
| outputs/rerun_main_baselines_civil.csv | `d2cf4bc30be375d0` | 86885 |
| outputs/rerun_main_baselines_civil_summary.csv | `3ef7b1686e14c24d` | 1142 |
| outputs/civil_main_table.csv | `daef0619b4989e80` | 735 |
| outputs/rerun/concat_control_github.csv | `f609b99dbb02053a` | 812 |
| outputs/rerun/concat_control_agnews.csv | `7531fd97fea79b40` | 813 |
| outputs/rerun/concat_control_civil.csv | `e4d11d830cc8e26f` | 819 |
| outputs/rerun/civil_title_archival.csv | `d2da6bff6b63b441` | 1379 |
| outputs/civil_openalex_topics.csv | `b35f870b0ba3716d` | 1097978 |
| outputs/civil_openalex_topics_meta.json | `887bef9e537b0aac` | 626 |
| outputs/rerun/civil_topic_cases_20261014.md | `000eb3cb99a5db6d` | 4970 |
| outputs/rerun/concat_fix_numbers_20261014.txt | `ece24eefb26dab71` | 1493 |

## code

| path | sha256 | bytes |
|---|---|---|
| config.py | `c67be6f0a0e7c00a` | 2072 |
| metrics.py | `195fc4e5f503308b` | 8466 |
| public_data.py | `840f24d23aed8b3b` | 9112 |
| public_data_github.py | `a196448511c394fa` | 11474 |
| models.py | `a5e69f7edf0a2e16` | 71200 |
| baselines.py | `255d72ea1a40b962` | 18423 |
| new_baselines.py | `b349935350c8f5e7` | 13914 |
| experiment.py | `d06fc42b4905ab92` | 16914 |
| benchmark_public.py | `622a91c0f5b8be1b` | 18149 |
| run_fcntm_train.py | `e0bb2f6621dd32d3` | 26765 |
| run_public_missing_field.py | `8308b303e87a68aa` | 30425 |
| occlusion_robustness.py | `a299ec9031339405` | 22978 |
| fair_readout.py | `9e06a2dc4e626cb3` | 6224 |
| rescore_clean_test.py | `ae2ab2fe45e45ecf` | 4177 |
| dump_btm_readout_topics.py | `c678c8b79734876a` | 3059 |
| add_btm_doc_topic.py | `c3ad08a14015a3a1` | 3777 |
| recompute_irbo.py | `163d972d41a4cb5b` | 2925 |
| run_ablation_public.py | `3d7459f602006ef2` | 18728 |
| run_core_ablations.py | `eeb83e60dc74590d` | 7045 |
| embed_public.py | `905c0c1fe0075054` | 10775 |
| public_data_civil.py | `4d25d029123d366f` | 13514 |
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
| field_complementarity.py | `8715ec5c51ae7832` | 6510 |
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
| verify_newseed_archives.py | `4e471727bda464b4` | 7397 |
| repin_fieldview_archive.py | `ecd4628018a3a746` | 5476 |
| finish_fieldview_regen.py | `11e11f585d89fd66` | 1653 |
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
| main.tex | `4f88b71ba48d43db` | 103274 |
| supplementary.tex | `60115bcc45242d00` | 19597 |
| main_en.tex | `e6cab35b46f442b8` | 84011 |
| supplementary_en.tex | `61c2558ea1352813` | 41293 |
| main_asce.tex | `ae54e16b3a8ef1b1` | 53991 |
| main_asce.bib | `1277149ea30a8f93` | 14367 |
| ascelike-new.cls | `b9002dae251b5666` | 25540 |
| ascelike-new.bst | `018056e4c6245316` | 34611 |
| topic_table_github_seed11_paper.tex | `9332105794211ccb` | 1331 |
| fig_main_comparison.pdf | `7021ad9a168d02a6` | 25957 |
| fig_occlusion.pdf | `7414f63a018c716f` | 34049 |
| fig_occlusion_cross_corpus.pdf | `27d12812ae7a2be2` | 44187 |
| fig_complementarity.pdf | `e536447760a853c4` | 40578 |
| fig_architecture_en.pdf | `9c564885ede64c13` | 87801 |
| fig_seed_paired_en.pdf | `437a77cdb3eb69c9` | 27172 |
