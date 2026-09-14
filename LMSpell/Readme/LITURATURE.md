# Literature Review of LLM based Spell Correction

## I. PLMS FOR SPELL CORRECTION

Table I contains a (nearly) comprehensive list of research that employed PLMs for spell correction. 47% focused on Chinese and 18% on English, both are HRLs [1]. In fact, only 7 studies worked on LRLs that fall into Category 3 or lower (which is fewer than the number of studies conducted on English). From the work on LRLs Sinhala [2], [3] and Bengali [4], [5] stand out as the languages for which, multiple attempts have been made. A majority of the efforts seem to be using BERT or its various derivatives.

### TABLE I
### RESEARCH THAT USED PLMS FOR SPELL CORRECTION

| Paper | Language - Category | Models Used |
| :--- | :--- | :--- |
| [6] | Chinese - 5 | BERT |
| [7] | English - 5 | BERT |
| [8] | English - 5 | GloVe, fastText, ELMo, GPT, GPT-2, BERT, ROBERTA, XLM-ROBERTa, BART, T5, XLNet |
| [9] | Sinhala - 2 | BERT |
| [10] | Chinese - 5 | BERT |
| [11] | English - 5 | BERT |
| [12] | Croatian - 3 | BERT |
| [13] | Vietnamese - 4 | BERT |
| [14] | English - 5 | DistilBERT, XLM-mBERT |
| [2] | Sinhala - 3 | ROBERTa, mT5, mBART |
| [4] | Bengali - 3 | BERT |
| [15] | Chinese - 5 | BERT |
| [16] | Chinese - 5 | BERT |
| [17] | Chinese - 5 | BERT |
| [18] | Chinese - 5 | GPT-3.5-Turbo |
| [19] | Chinese - 5 | BERT, BART, BAICHUAN-13B-Chat, Text-Davinci-GPT-3.5 |
| [20] | Thai - 3 | WangchanBERTa |
| [21] | Kannada - 3 | mT5 |
| [22] | Russian - 4, English - 5 | BERT, T5, GPT-3.5, GPT-4.0, M2M-100 |
| [23] | Chinese - 5 | BERT, BART |
| [24] | Chinese - 5 | BERT |
| [25] | Chinese - 5 | BERT, BAICHUAN, GPT2 |
| [26] | Chinese - 5 | BERT, ChineseBERT |
| [27] | Chinese - 5 | BERT, GPT-3.5, BAICHUAN |
| [28] | Chinese - 5 | BERT, C-LLM, GPT-4 |
| [29] | English - 5 | BART, T5 |
| [30] | English - 5 | BERT |
| [31] | English - 5 | BERT, Mistral-7B, Claude-3-sonnet |
| [32] | Persian - 4 | BERT |
| [33] | Turkish - 4 | TURNA |
| [5] | Bengali - 3 | BERT |
| [34] | Chinese - 5 | BERT, BAICHUAN2, Qwen2.5, InternLM2.5 |
| [35] | Chinese - 5 | BERT, GPT-3.5, GLM4 |
| [36] | Hindi - 3 | IndicBert, mT5, mBart |
| [37] | Chinese - 5 | BERT, BAICHUAN2, Qwen2.5, InternL |
| [38] | Turkish - 4 | BERT, XLM-ROBERTa, T5 |
| [39] | Turkish - 4 | DistilBERT |
| [40] | Indonesian - 4 | T5 Base, Bart |
| [41] | Chinese - 5 | ROBERTA, BERT-wwm, MacBERT |

*Language category based on [1]*

## REFERENCES

[1] S. Ranathunga and N. de Silva, "Some languages are more equal than others: Probing deeper into the linguistic disparity in the NLP world." in Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), Y. He. H. Ji, S. Li, Y. Liu, and C.-H. Chang, Eds. Online only: Association for Computational Linguistics, Nov. 2022, pp. 823-848. [Online]. Available: https://aclanthology.org/2022.aacl-main.62/

[2] P. Sudesh, D. Dashintha, R. Lakshan, and G. Dias, "Erroff: A tool to identify and correct real-word errors in sinhala documents." in 2022 Moratuwa Engineering Research Conference (MERCon). IEEE, 2022, pp. 1-6.

[3] C. Sonnadara, S. Ranathunga, and S. Jayasena, "Sinhala spell correction: A novel benchmark with neural spell correction," ResearchGate, 2021.

[4] C. Rahman, M. Rahman, S. Zakir, M. Rafsan, and M. E. Ali, "BSpell: A CNN-blended BERT based Bangla spell checker," in Proceedings of the First Workshop on Bangla Language Processing (BLP-2023), F. Alam, S. Kar, S. A. Chowdhury, F. Sadeque, and R. Amin, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 7-17. [Online]. Available: https://aclanthology.org/2023.banglalp-1.2/

[5] D. Banik, S. Das, S. Martha, and A. Shankar, "BERT-Inspired Progressive Stacking to Enhance Spelling Correction in Bengali Text," ACM Transactions on Asian and Low-Resource Language Information Processing, vol. 23, no. 8. pp. 1-12, 2024.

[6] X. Cheng, W. Xu, K. Chen, S. Jiang. F. Wang, T. Wang, W. Chu, and Y. Qi, "SpellGCN: Incorporating phonological and visual similarities into language models for Chinese spelling check," in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Association for Computational Linguistics, Jul. 2020, pp. 871-881. [Online]. Available: https://aclanthology.org/2020.acl-main.81/

[7] S. Zhang, H. Huang, J. Liu, and H. Li, "Spelling error correction with soft-masked BERT," in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, Jul. 2020, pp. 882-890. [Online]. Available: https://aclanthology.org/2020.acl-main.82/

[8] J.-H. Lee, M. Kim, and H.-C. Kwon, "Deep learning-based context-sensitive spelling typing error correction," IEEE Access, vol. 8, pp. 152565-152578, 2020.

[9] Y. Hu, X. Jing, Y. Ko, and J. T. Rayz, "Misspelling correction with pre-trained contextual language model," in 2020 ieee 19th international conference on cognitive informatics & cognitive computing (icci* cc). IEEE, 2020, pp. 144-149.

[10] S. M. Jayanthi, D. Pruthi, and G. Neubig, "NeuSpell: A neural spelling correction toolkit," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, Q. Liu and D. Schlangen, Eds. Online: Association for Computational Linguistics, Oct. 2020, pp. 158-164. [Online]. Available: https://aclanthology.org/2020.emnlp-demos.21/

[11] S. Liu, T. Yang, T. Yue, F. Zhang, and D. Wang, "PLOME: Pre-training with misspelled knowledge for Chinese spelling correction," in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), C. Zong. F. Xia, W. Li, and R. Navigli, Eds. Online: Association for Computational Linguistics, Aug. 2021. pp. 2991-3000. [Online]. Available: https://aclanthology.org/2021.acl-long.233/

[12] F. Tohidian, A. Kashiri, and F. Lotfi, "Bedspell: Spelling error correction using bert-based masked language model and edit distance." in International Conference on Service-Oriented Computing. Springer, 2022, pp. 3-14.

[13] T. H. Ngo, H. D. Tran, T. Huynh, and K. Hoang, "A combination of bert and transformer for vietnamese spelling correction," in Asian Conference on Intelligent Information and Database Systems. Springer, 2022, pp. 545-558.

[14] M. Mitreska, K. Mishev, and M. Simjanoska, "NIp-based typo correction model for croatian language," in 2022 45th Jubilee International Convention on Information, Communication and Electronic Technology (MIPRO). IEEE, 2022, pp. 942-947.

[15] X. Wei, J. Huang, H. Yu, and Q. Liu, "PTCSpell: Pre-trained corrector based on character shape and Pinyin for Chinese spelling correction," in Findings of the Association for Computational Linguistics: ACL 2023, A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 6330-6343. [Online]. Available: https://aclanthology.org/2023.findings-acl.394/

[16] H. Wu, S. Zhang, Y. Zhang, and H. Zhao, "Rethinking masked language modeling for Chinese spelling correction," in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 10743-10756. [Online]. Available: https://aclanthology.org/2023.acl-long.600/

[17] H. Huang, J. Ye, Q. Zhou, Y. Li Y. Li F. Zhou, and H.-T. Zheng. "A frustratingly easy plug-and-play detection-and-reasoning module for Chinese spelling check," in Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 11514-11525. [Online]. Available: https://aclanthology.org/2023.findings-emnlp.771/

[18] X. Zhang, X. Zhang, C. Yang, H. Yan, and X. Qiu, "Does correction remain a problem for large language models?" arXiv preprint arXiv:2308.01776, 2023.

[19] Y. Li, H. Huang, S. Ma, Y. Jiang, Y. Li, F. Zhou, H.-T. Zheng, and Q. Zhou, "On the (in) effectiveness of large language models for chinese text correction," arXiv preprint arXiv:2307.09007, 2023.

[20] I. Pankam, P. Limkonchotiwat, and E. Chuangsuwanich, "Two-stage thai misspelling correction based on pre-trained language models," in 2023 20th international joint conference on computer science and software engineering (jesse). IEEE, 2023, pp. 7-12.

[21] S. Ramaneedi and P. B. Pati, "Kannada textual error correction using t5 model," in 2023 IEEE 8th International Conference for Convergence in Technology (I2CT). IEEE, 2023, pp. 1-5.

[22] N. Martynov, M. Baushenko, A. Kozlova, K. Kolomeytseva, A. Abramov, and A. Fenogenova, "A methodology for generative spelling correction via natural spelling errors emulation across multiple domains and languages," in Findings of the Association for Computational Linguistics: EACL 2024, Y. Graham and M. Purver, Eds. St. Julian's, Malta: Association for Computational Linguistics, Mar. 2024, pp. 138-155. [Online]. Available: https://aclanthology.org/2024.findings-eacl.10/

[23] J. Su, Y. Xie, and Y. Mou, "Ucsc-cgec: A unified approach for chinese spelling check and grammatical error correction," in 2024 International Joint Conference on Neural Networks (IJCNN). IEEE, 2024. pp. 1-8.

[24] S. Wang, C. Tong, K. Peng, and L. Jiang, "Local attention augmentation for chinese spelling correction," in International Conference on Computational Science. Springer, 2024, pp. 438-452.

[25] L. Liu, H. Wu, and H. Zhao, "Chinese spelling correction as rephrasing language model," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024, pp. 18662-18 670.

[26] H. Wu, H. Zhang, R. Xuan, and D. Song. "Bi-DCSpell: A bi-directional detector-corrector interactive framework for Chinese spelling check," in Findings of the Association for Computational Linguistics: EMNLP 2024, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 3974-3984. [Online]. Available: https://aclanthology.org/2024.findings-emnlp.229/

[27] L. Jiang. H. Wu, H. Zhao, and M. Zhang, "Chinese spelling corrector is just a language learner," in Findings of the Association for Computational Linguistics: ACL 2024, L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 6933-6943. [Online]. Available: https://aclanthology.org/2024.findings-acl.413/

[28] K. Li, Y. Hu, L. He, F. Meng, and J. Zhou, "C-LLM: Learn to check Chinese spelling errors character by character," in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 5944-5957. [Online]. Available: https://aclanthology.org/2024.emnlp-main.340/

[29] A. Dutta, G. Polushin, X. Zhang, and D. Stein, "Enhancing e-commerce spelling correction with fine-tuned transformer models," in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 4928-4938.

[30] J. Wen, X. Feng, and F. Fu, "English text spelling error detection and correction based on multi-feature data fusion algorithm," in 2024 International Conference on Distributed Computing and Optimization Techniques (ICDCOT). IEEE, 2024, pp. 1-5.

[31] X. Guo, R. Patki, D. Everaert, and C. Potts, "Retrieval augmented spelling correction for E-commerce applications," in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, F. Dernoncourt, D. Preoțiuc-Pietro. and A. Shimorina, Eds. Miami, Florida, US: Association for Computational Linguistics, Nov. 2024, pp. 73-79. [Online]. Available: https://aclanthology.org/2024.emnlp-industry.7/

[32] A. Naziri and H. Zeinali, "A comprehensive approach to misspelling correction with bert and levenshtein distance." arXiv preprint arXiv:2407.17383, 2024.

[33] D. Senturk, M. B. Topal, S. Adiguzel, M. Ozturk, and A. Basar, "Spelling corrector for turkish product search," in 2024 34th International Conference on Collaborative Advances in Software and Computing (CASCON), 2024, pp. 1-8.

[34] H. Zhou, Z. Li, B. Zhang, C. Li, S. Lai, J. Zhang, F. Huang, and M. Zhang, "A simple yet effective training-free prompt-free approach to Chinese spelling correction based on large language models," in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 17446-17467. [Online]. Available: https://aclanthology.org/2024.emnlp-main.966/

[35] M. Dong, Z. Cheng, C. Luo, and T. He, "Retrieval-augmented generation for large language model based few-shot Chinese spell checking," in Proceedings of the 31st International Conference on Computational Linguistics, O. Rambow, L. Wanner, M. Apidianaki, H. Al-Khalifa, B. D. Eugenio, and S. Schockaert, Eds. Abu Dhabi, UAE: Association for Computational Linguistics, Jan. 2025, pp. 10767-10780. [Online]. Available: https://aclanthology.org/2025.coling-main.717/

[36] S. K. Behera and R. Saluja, "Hilearners: Non-native spoken hindi error correction," in Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, 2025, pp. 1276-1288.

[37] Z. Qiao, H. Zhou, and Z. Li, "Mixture of small and large models for chinese spelling check," in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 28 298-28 311.

[38] C. G. Turhan, "Leveraging large language models for spelling correction in turkish," Peer Computer Science, vol. 11, p. e2889, 2025.

[39] M. Ülker and A. B. Özer, "A novel model of deep learning-based spelling detection in turkish," in 2025 10th International Conference on Computer Science and Engineering (UBMK), 2025, pp. 1558-1562.

[40] M. Lefrandt, E. B. Santoso, A. A. S. Gunawan, and J. J. Tedjasulaksana. "Contextual spelling corrector for indonesian text preprocessing: A comparative analysis of large language models," in 2025 IEEE International Conference on Industry 4.0, Artificial Intelligence, and Communications Technology (IAICT), 2025, pp. 290-296.

[41] L. Zhang, Z. Liu, Q. Yan, and X. Liu, "Psif: Phonetic-semantic and long-short information fusion for chinese spelling correction," Applied Sciences, vol. 16, no. 5, 2026. [Online]. Available: https://www.mdpi.com/2076-3417/16/5/2440
LMSpell_Mercon.md
Displaying LMSpell_Mercon.md.
