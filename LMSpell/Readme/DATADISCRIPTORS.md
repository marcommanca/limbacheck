### Languages Used

| Language | Dataset | Number of Sentences: Train | Number of Sentences: Val. | Number of Sentences: Test | Family | Resource Level* | PLMs** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Azerbaijani | LocalDoc, 2024 | 81,176 | 1,000 | 2,000 | Turkic | Low (Category 3) | mT5, XLM-R |
| Bulgarian | Klouchek & Batista-Navarro, 2024 | 20,719 | 1,000 | 2,000 | Indo-European (Slavic) | Low (Category 3) | mT5, XLM-R |
| French | rasaboun, 2022; fdemelo, 2022 | 5,393 | 500 | 1,000 | Indo-European (Romance) | High (Category 5) | mT5, mBART50, XLM-R, Llama 3.1 |
| Hindi | Etoori et al., 2018 | 160,000 | 1,000 | 2,000 | Indo-European (Indo-Aryan) | Low (Category 3) | mT5, mBART50, XLM-R, Llama 3.1 |
| Korean | vitruv, 2022 | 77,000 | 1,000 | 2,000 | Koreanic | High (Category 4) | mT5, mBART50, XLM-R |
| Sinhala | Sonnadara et al., 2021; Sudesh et al., 2022 | 510,706 | 5,282 | 2,037 | Indo-European (Indo-Aryan) | Low (Category 2) | mT5, mBART50, XLM-R |
| Vietnamese | Ngo et al., 2022 | 4,500 | 500 | 1,000 | Austroasiatic | High (Category 4) | mT5, XLM-R |
| Indonesian | Yanfi et al., 2023 | 387,000 | - | - | Austronesian | High (Category 4) | mT5, mBART50, XLM-R |
| Turkish | Turhan, 2025 | 687,008 | 85,871 | 85,824 | Turkic | High (Category 4) | mT5, mBART50, XLM-R |

**Resource level is according to Ranathunga & de Silva, 2022's language categorization.*

*PLMs that were pre-trained on data from each language.*

*Note: The Gemma 3 family of models supports over 140 languages, but does not explicitly specify the complete list.*

---

The table above shows the details of the languages used in this study. Given the wide-spread nature of the Indo-European (IE) language family, we have opted to also show in parenthesis the subtree to which IE languages belong. We have attempted to get a good spread of languages where they cover the range of Category 2 up to Category 5 by the definition provided by Ranathunga & de Silva, 2022. Due to the even lower resource nature of the languages, we were unable to find any suitable datasets from Category 0 or Category 1. Dataset descriptions are given below. Note that not all datasets are associated with a dataset description. We could only find details for Bulgarian and Vietnamese.

* **Bulgarian (Klouchek & Batista-Navarro, 2024)** - The dataset was derived from Bulgarian Wikipedia articles. To filter out noisy content such as titles, links, and fragments, only sentences containing at least three words and at least one verb were retained. The dataset was then constructed by automatically introducing spelling errors into otherwise correct sentences using Python scripts. For each selected sentence, the original form was kept as the reference, an error of a predefined type was induced whenever the sentence satisfied the relevant linguistic conditions, and the correct and erroneous versions were paired. Errors were applied only to lemmas longer than three characters and excluding capitalized words, to avoid function words and probable named entities. The resulting data covers seven error types: vowel change, double consonant alternation, end-of-lemma consonant alternation, deletion of repeated *t* or *n*, loss of *t* or *d* sounds, random character substitution, and semantic errors that preserve syntactic validity while disrupting sentence meaning.
* **Vietnamese (Ngo et al., 2022)** - The dataset has been sourced from a corpus that contains articles from multiple news outlets. After pre-processing the dataset, common spelling error types such as fat finger, region-based errors and telex errors (since Vietnamese uses a derivation of the Latin script) have been randomly introduced to the dataset.

---

### References

* LocalDoc, “azerbaijani spelling (revision c6bd258),” 2024. [On-line]. Available: [https://huggingface.co/datasets/LocalDoc/azerbaijani_spelling](https://huggingface.co/datasets/LocalDoc/azerbaijani_spelling)
* B. Klouchek and R. Batista-Navarro, “Bulgarian grammar error correction with data augmentation and machine translation techniques,” in *Proceedings of the 7th International Conference on Natural Language and Speech Processing (ICNLSP 2024)*, M. Abbas and A. A. Freihat, Eds. Trento: Association for Computational Linguistics, Oct. 2024, pp. 365–376. [Online]. Available: [https://aclanthology.org/2024.icnlsp-1.38/](https://aclanthology.org/2024.icnlsp-1.38/)
* rasaboun, “Spelling correction french,” 2022. [Online]. Available: [https://huggingface.co/datasets/rasaboun/spellingcorrectionFrench](https://huggingface.co/datasets/rasaboun/spellingcorrectionFrench) (accessed: 2025-05-16).
* fdemelo, “Spelling correction french news,” 2022. [Online]. Available: [https://huggingface.co/datasets/fdemelo/spelling-correction-french-news]() (accessed: 2025-05-16).
* P. Etoori, M. Chinnakotla, and R. Mamidi, “Automatic spelling correction for resource-scarce languages using deep learning,” in *Proceedings of ACL 2018, Student Research Workshop*, V. Shwartz, J. Tabassum, R. Voigt, W. Che, M.-C. de Marneffe, and M. Nissim, Eds. Melbourne, Australia: Association for Computational Linguistics, Jul. 2018, pp. 146–152. [Online]. Available: [https://aclanthology.org/P18-3021/]()
* vitruv, “Err spelling korean,” 2022. [Online]. Available: [https://huggingface.co/datasets/vitruv/errspelling_kor]() (accessed: 2025-05-16).
* P. Sudesh, D. Dashintha, R. Lakshan, and G. Dias, “Erroff: A tool to identify and correct real-word errors in sinhala documents,” in *2022 Moratuwa Engineering Research Conference (MERCon)*. IEEE, 2022, pp. 1–6.
* C. Sonnadara, S. Ranathunga, and S. Jayasena, “Sinhala spell correction: A novel benchmark with neural spell correction,” *ResearchGate*, 2021.
* T. H. Ngo, H. D. Tran, T. Huynh, and K. Hoang, “A combination of bert and transformer for vietnamese spelling correction,” in *Asian Conference on Intelligent Information and Database Systems*. Springer, 2022, pp. 545–558.
* Y. Yanfi, R. Setiawan, H. Soeparno, and W. Budiharto, “Specil: Spell error corpus for the indonesian language,” *IEEE Access*, vol. 11, pp. 93,227–93,237, 2023.
* C. G. Turhan, “Leveraging large language models for spelling correction in turkish,” *PeerJ Computer Science*, vol. 11, p. e2889, 2025.
* S. Ranathunga and N. de Silva, “Some languages are more equal than others: Probing deeper into the linguistic disparity in the NLP world,” in *Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, Y. He, H. Ji, S. Li, Y. Liu, and C.-H. Chang, Eds. Online only: Association for Computational Linguistics, Nov. 2022, pp. 823–848. [Online]. Available: [https://aclanthology.org/2022.aacl-main.62/]()