# Verified related-work notes

Verification sources: official proceedings pages (PMLR, proceedings.neurips.cc, openaccess.thecvf.com, iclr.cc), arXiv abstracts and full texts (read locally). DBLP is behind a bot challenge and Semantic Scholar rate-limited, so neither was used. OpenReview was also blocked; ICLR venue facts come from iclr.cc pages and arXiv comments ("Accepted by ICLR ...").

## Already in references.bib (do not duplicate)
chen2020simclrv2, khosla2020supcon, assran2020suncet, oliver2018realistic, he2020moco, dacosta2022sololearn, galanti2022role, hui2022limitations, graf2021dissecting, yang2020rethinking, kumar2022lpft, islam2021broad, chen2021intriguing, cubuk2020randaugment (CVPRW). Existing entries are correct but lack volume/pages; updated versions are in related.bib under the same keys. Items 1-10 etc. are new.

## Corrections to the reviewer's details
- #32 Galanti paper: authors are Galanti, **Gyorgy (Andras), Hutter (Marcus)**, not "George, Hacohen". The existing galanti2022role entry is correct.
- #36 RandAugment: both exist. CVPRW 2020 (pp. 702-703, short version) AND NeurIPS 2020 (vol. 33, pp. 18613-18624, full). Existing key cites the CVPRW one; not wrong, but NeurIPS is the full paper.
- #6 Henaff: the PMLR page lists only "Olivier Henaff" (a PMLR metadata quirk). The real author list (arXiv 1905.09272v3) has 7 authors; the bib uses the full list.
- #26 Efficiency Misnomer: author order differs between arXiv v2 and the ICLR-published PDF (Dehghani, Arnab, Beyer, Vaswani, Tay in the PDF header, equal-contribution marks on Dehghani, Arnab, Beyer, Tay) and the iclr.cc listing (Dehghani, Tay, Arnab, Beyer, Vaswani). Used the PDF order; low-stakes.
- #35 Le & Yang: first author is Ya Le (second Xuan S. Yang); "Le & Yang" is fine. No primary PDF found; see unverified.
- #38 Assran SuNCEt: not a main-conference paper; it is a NeurIPS 2020 Workshop on Self-Supervised Learning paper (arXiv 2006.10803v2). Cite as workshop.
- #37 Hui et al.: no peer-reviewed version found (arXiv 2202.08384v1 only); keep as arXiv.
- #20 Liu et al.: ICLR 2022 (spotlight per iclr.cc) is right; an earlier version was a NeurIPS 2021 workshop paper (so some indexes show 2021).
- #39 solo-learn: the paper's table does NOT say ResNet-18 / 1000 epochs, and SupCon is NOT in it (see below).
- #19, #21, #22 etc.: venues and years confirmed as given.

## Notes per paper

**chen2022perfectly** (ICML 2022, PMLR 162:3090-3122; arXiv 2204.07596). Authors Mayee F. Chen, Daniel Y. Fu, Avanika Narayan, Michael Zhang, Zhao Song, Kayvon Fatahalian, Christopher Re. Class collapse = all points of a class map to one representation; SupCon exhibits it on training data, which hurts coarse-to-fine transfer and worst-group robustness. They prove a class-collapsed geometry cannot be optimal for spread, and that adding a weighted class-conditional InfoNCE term controls spread: L = (1-alpha) L_sup + alpha L_cNCE, where L_cNCE has the instance's own augmentation as the positive and the other same-class points as negatives (Eq. 3 in paper). So it is essentially an instance-discrimination term restricted to within-class negatives, very close to adding SimCLR-style instance term to SupCon (ours mixes labelled and unlabelled differently, but the mechanism is the same). Their full method (THANOS) adds a class-conditional autoencoder; reported +11.1 points on coarse-to-fine transfer over 5 datasets, +4.7 worst-group points over 3 datasets, SOTA on CelebA by 11.5 (abstract). Their experiments use coarse labels and full label sets, not a 10% label budget. Source: https://proceedings.mlr.press/v162/chen22d.html ; https://arxiv.org/abs/2204.07596

**kornblith2021better** (NeurIPS 2021, vol 34, pp 28648-28662). Authors Kornblith, Chen, Lee, Norouzi. Objectives that beat softmax cross-entropy on ImageNet give fixed feature extractors that transfer substantially worse; with full fine-tuning the loss matters little. Link: higher penultimate-layer class separation helps the original task and hurts transfer. Relevant to why supervised-heavy pretraining may transfer worse. Source: https://papers.nips.cc/paper/2021/hash/f0bf4a2da952528910047c31b6c2e951-Abstract.html ; arXiv 2010.16402.

**xue2023which** (ICML 2023, PMLR 202:38938-38970; arXiv 2305.16536). Authors Xue, Joshi, Gan, Chen, Mirzasoleiman. Unified theory of class collapse (supervised CL collapses subclasses) and feature suppression (unsupervised CL suppresses harder class-relevant features); SGD simplicity bias is the key factor; remedies they motivate are larger embedding dimension and better augmentation. Source: https://proceedings.mlr.press/v202/xue23d.html

**zhao2021what** (ICLR 2021; arXiv 2006.06606, comment "Accepted by ICLR 2021"). Zhao, Wu, Lau, Lin: instance discrimination transfers via low/mid-level features, and the intra-category invariance enforced by supervised pretraining hurts transfer; they also study how to exploit labels better for pretraining. No page numbers (ICLR). Sources: https://arxiv.org/abs/2006.06606 ; https://iclr.cc/virtual/2021/session/4307 (title listed). Note: PMLR v119/zhao20a is a different paper.

**zoph2020rethinking** (NeurIPS 2020, vol 33, pp 3833-3845). Zoph, Ghiasi, Lin, Cui, Liu, Cubuk, Le. COCO detection: stronger augmentation and more labeled data diminish the value of ImageNet pre-training; self-training helps in all regimes and is additive with pre-training. Source: https://proceedings.neurips.cc/paper/2020/hash/27e9661e033a73a6ad8cefcde965c54d-Abstract.html

**henaff2020data** (ICML 2020, PMLR 119:4182-4192; arXiv 1905.09272). CPC v2 features allow 2-5x fewer labels than pixel-trained classifiers on ImageNet and beat supervised pretraining for VOC detection transfer. Sources: https://proceedings.mlr.press/v119/henaff20a.html ; https://arxiv.org/abs/1905.09272 (full author list).

**hacohen2022active** (ICML 2022, PMLR 162:8175-8195; TypiClust). Hacohen, Dekel, Weinshall. Phase-transition view: at low budgets query typical (dense-region) examples, at high budgets query hard/unrepresentative ones; TypiClust clusters self-supervised embeddings and picks the most typical point per cluster; improves strongly in the low-budget regime and further when combined with semi-supervised learning. Source: https://proceedings.mlr.press/v162/hacohen22a.html

**yehuda2022active** (NeurIPS 2022, vol 35, pp 22354-22367; ProbCover, arXiv 2205.11320). Yehuda, Dekel, Hacohen, Weinshall. Subset selection via maximizing probability coverage of balls in self-supervised embedding space; improves low-budget AL, with dual view yielding Coreset-like high-budget strategies. Note first author is Ofer Yehuda. Source: https://proceedings.neurips.cc/paper_files/paper/2022/hash/8c64bc3f7796d31caa7c3e6b969bf7da-Abstract-Conference.html

**sorscher2022beyond** (NeurIPS 2022 main track, vol 35, pp 19523-19536). Sorscher, Geirhos, Shekhar, Ganguli, Morcos. In theory a good pruning metric gives better-than-power-law scaling, with the best strategy depending on data size: keep easy examples when data is scarce, hard when abundant; confirmed on ResNets on CIFAR-10, SVHN, ImageNet; benchmarks ten pruning metrics on ImageNet and proposes a self-supervised metric. Source: https://proceedings.neurips.cc/paper_files/paper/2022/hash/7b75da9b61eda40fa35453ee5d077df6-Abstract-Conference.html

**zheng2023coverage** (ICLR 2023; arXiv 2210.15809). Zheng, Liu, Lai, Prakash. At high pruning rates importance-based coreset selection falls below random; coverage matters; CCS stratifies by importance to keep coverage. Abstract (arXiv v2) says gains at 90% pruning on CIFAR-10 over prior SOTA; the figure differs between versions (a search summary quoted 7.04 vs random and >=20.16 vs importance methods; arXiv abstract says >=19.56), so check the PDF before quoting numbers. Sources: https://arxiv.org/abs/2210.15809 ; https://iclr.cc/virtual/2023/poster/11710

**li2021comatch** (ICCV 2021, pp 9475-9484; arXiv 2011.11183). Li, Xiong, Hoi. Joint class-probability and embedding learning with graph-based contrastive regularization; ImageNet 1% labels 66.0% top-1 (+12.6 over FixMatch per abstract); large gains on label-scarce CIFAR-10 and STL-10. Source: openaccess.thecvf.com ICCV2021 CoMatch page.

**zheng2022simmatch** (CVPR 2022, pp 14471-14481; arXiv 2203.06915). Zheng, You, Huang, Wang, Qian, Xu. Consistency on both semantic and instance similarity, labeled memory buffer; ImageNet 400 epochs: 67.2% (1% labels) and 74.4% (10%) top-1. Source: openaccess.thecvf.com CVPR2022 SimMatch page.

**yang2022classaware** (CVPR 2022, pp 14421-14430; CCSSL, arXiv 2203.02261). Yang, Wu, Zhang, Jiang, Liu, Zheng, Zhang, Wang, Zeng (9 authors). Drop-in helper for pseudo-label SSL: class-wise contrastive on reliable in-distribution data, image-wise contrastive on OOD data, target re-weighting; on Semi-iNat 2021 improves FixMatch by 9.80% and CoMatch by 3.18%.

**fini2023semi** (CVPR 2023, pp 3187-3197; arXiv 2306.07483). Fini, Astolfi, Alahari, Alameda-Pineda, Mairal, Nabi, Ricci. Turns clustering SSL (SwAV/DINO) into a semi-supervised learner with one cross-entropy loss combining labeled and cluster assignments (centroids as class prototypes); SOTA on CIFAR-100 and ImageNet at the time. Same group as solo-learn.

**wang2022usb** (NeurIPS 2022 Datasets and Benchmarks, vol 35, pp 3938-3961; arXiv 2208.07204). 22 authors (first Yidong Wang). 15 classification tasks over CV/NLP/audio; uses pretrained models to cut cost (FixMatch across tasks ~39 V100-days vs ~335 for 5 CV tasks in TorchSSL, per the search summary of the proceedings page).

**li2020budgeted** (ICLR 2020; arXiv 1905.04753). Li, Yumer, Ramanan. Under a fixed iteration budget the LR schedule must be budget-aware; linear decay to zero is robust and best among tested; "budgeted convergence". Directly relevant to compute-matched comparisons. Sources: https://iclr.cc/virtual_2020/poster_HyxLRTVKPH.html

**chen2021intriguing** (NeurIPS 2021, vol 34, pp 11834-11845). Generalized contrastive loss family performs similarly with a non-linear projection head; instance-based CL on multi-object images; feature suppression. Verified correct in existing bib.

**robinson2021can** (NeurIPS 2021, vol 34, pp 4974-4986; arXiv 2106.11230). Robinson, Sun, Yu, Batmanghelich, Jegelka, Sra. Instance discrimination difficulty controls which features are extracted; harder pairs improve some features and suppress others (shortcuts); propose implicit feature modification. Relevant: mixing in labels (class positives) changes instance task difficulty.

**xiao2021what** (ICLR 2021; arXiv 2008.05659, "Published as a conference paper at ICLR 2021"). Xiao, Wang, Efros, Darrell. Separate embedding spaces each invariant to all but one augmentation; concatenation helps coarse/fine-grained/few-shot tasks.

**liu2022self** (ICLR 2022; arXiv 2110.05025). Liu, HaoChen, Gaidon, Ma. SSL representations are more robust to class imbalance than supervised; SSL learns label-irrelevant features from frequent classes; proposes re-weighted SAM regularizer.

**yang2020rethinking** (NeurIPS 2020, vol 33, pp 19290-19301). Yang & Xu. Imbalanced labels still help with semi-supervised use; self-supervised pretraining then supervised training improves imbalanced learning.

**coates2011analysis** (AISTATS 2011, PMLR 15:215-223). Coates, Ng, Lee; STL-10 is introduced here. Source: https://proceedings.mlr.press/v15/coates11a.html

**chen2020mocov2** (arXiv 2003.04297, tech report, no peer-reviewed version). Chen, Fan, Girshick, He. MLP head and stronger augmentation in MoCo outperform SimCLR without large batches. (The 71.1% linear number was not verified.)

**he2020moco** (CVPR 2020, pp 9729-9738). Existing entry correct; pages added.

**oliver2018realistic** (NeurIPS 2018, vol 31; pages not on the official page, omitted; commonly cited as 3235-3246). Verified from full text: P.6 "Realistically Small Validation Sets" says the validation set is often much larger than the labeled training set, e.g. SVHN has ~7,000 validation examples while many papers use 1,000 training labels, i.e. over seven times bigger; in real applications it would be used for training; hyperparameter tuning on it is unrealistic. Sec 4.6: with a realistic validation set (10% of train size) models cannot be reliably told apart; recommendation: do not over-tweak hyperparameters on an unrealistically large validation set. Also: well-tuned supervised baselines are underreported; OOD unlabeled data can hurt. Source: arXiv 1804.09170 and https://proceedings.neurips.cc/paper/2018/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html

**dehghani2022efficiency** (ICLR 2022, arXiv 2110.12894). "The overall gist of the efficiency misnomer is that no single cost indicator is sufficient." Cost indicators (params, FLOPs, throughput/speed, memory) are commonly assumed correlated but can contradict one another; incomplete reporting gives partial conclusions; suggests reporting several and fixing cost across models for fair comparison.

**kumar2022lpft** (ICLR 2022 oral). Existing entry correct. Fine-tuning can distort pretrained features; LP-FT gets about +1% ID and +10% OOD over full fine-tuning (as summarized on iclr.cc/arXiv).

**newell2020useful** (CVPR 2020, pp 7345-7354). Newell & Deng. Synthetic-data study; benefit of SSL pretraining is largest with few labels and fades before performance plateaus as labels grow; linear evaluation does not correlate with fine-tuning performance.

**islam2021broad** (ICCV 2021, pp 8845-8855). Existing entry correct (authors Islam, Chen, Panda, Karlinsky, Radke, Feris).

**graf2021dissecting** (ICML 2021, PMLR 139:3821-3830). Existing entry correct.

**inoue2020semi** (APSIPA ASC 2020; arXiv 2006.04326). Inoue & Goto. Generalized contrastive loss (GCL) unifies supervised metric learning and unsupervised contrastive losses, so one loss handles supervised, semi-supervised and unsupervised training of speaker embeddings (VoxCeleb). Venue "APSIPA 2020" taken from secondary listings (AMiner), proceedings page not reached; pages omitted.

**galanti2022role** (ICLR 2022; arXiv 2112.15121). Verified from full text, Sec. 3: for a feature map f and two class-conditional distributions Q1, Q2, CDNV V_f(Q1,Q2) = (Var_f(Q1) + Var_f(Q2)) / (2 ||mu_f(Q1) - mu_f(Q2)||^2), where mu_f and Var_f are the mean and (total) variance of f(x), x~Q (Var is bounded by feature dimension p). Extended to finite sets via the uniform distribution on the set. It is a per-pair quantity; neural collapse at training is stated as lim_t Avg_{i != j in [l]} V_{f_t}(S_i, S_j) = 0, i.e. averaged over all ordered class pairs i != j. Also: nearest-class-mean error is bounded by 16(k-1) V_f(P1,P2)(1+1/n_c) (Prop. 5). Source: https://arxiv.org/pdf/2112.15121

**hui2022limitations**: only arXiv 2202.08384v1 (17 Feb 2022) found; no journal/conference version found. Main claim: neural collapse occurs on the train set but not on test.

**assran2020suncet**: workshop paper (NeurIPS 2020 Workshop on Self-Supervised Learning). Headline: SuNCEt (NCA/NCE-based loss using labels during pre-training) matches the semi-supervised accuracy of SimCLR on ImageNet "while using less than half the amount of pre-training and compute". Precisely (Sec. 4): with 10% labels, SimCLR+SuNCEt matches best SimCLR top-5 using 44% of compute and top-1 using 45%; with 1% labels only 81% (top-5) / 83% (top-1) of compute. Table 1 (ResNet-50, 10% labels): SimCLR 500 ep 65.4 top-1, +SuNCEt 500 ep 66.7; SimCLR 1000 ep 66.1 (their reimpl.). At equal epochs: +1.6 top-1 at 100 epochs (61.8 -> 63.4), +1.3 at 500 epochs. CIFAR-10 also evaluated (ResNet-50, 500 epochs default). They note overfitting on CIFAR-10 when SuNCEt stays on for the whole run. Source: arXiv 2006.10803v2.

**dacosta2022sololearn** (JMLR 23(56):1-6, 2022). The comparison is Table 1 (online linear evaluation, CIFAR-10 / CIFAR-100 / ImageNet-100); the paper text I read states neither "ResNet-18" nor "1000 epochs" (those details are in the repo, not verified here). CIFAR-10 Acc@1 over the 11 methods: min 88.67 (W-MSE), max 92.94 (MoCo V2+). SimCLR 90.74, BYOL 92.58, SimSiam 90.51. SupCon is listed among implemented methods but has no row in Table 1. (Others: Barlow Twins 92.10, VICReg 92.07, NNCLR 91.88, ReSSL 90.63, DINO 89.52, SwAV 89.17, DeepCluster V2 88.85.) Source: https://jmlr.org/papers/volume23/21-1155/21-1155.pdf

**chen2020simclrv2** (NeurIPS 2020, vol 33, pp 22243-22255; arXiv 2006.10029). Three steps: (1) unsupervised/self-supervised pretraining of a big ResNet with SimCLRv2 (deeper projection head, larger model, memory), (2) supervised fine-tuning on the few labels (from a middle layer of the projection head), (3) distillation on unlabeled data (self-training with the fine-tuned model as teacher, into the same or a smaller student). ResNet-50 student: 73.9% top-1 with 1% labels and 77.5% with 10% labels (supervised ResNet-50 with all labels: 76.6%). Largest (ResNet-152 3x+SK) with distillation: 76.6% / 80.9% for 1% / 10%. Linear eval 79.8%.

**khosla2020supcon** (NeurIPS 2020, vol 33, pp 18661-18673). Table 2 (ResNet-50, top-1): CIFAR-10 SimCLR 93.6, cross-entropy 95.0, SupCon 96.0 (CIFAR-100: 70.7 / 75.3 / 76.5; ImageNet 70.2 / 78.2 / 78.7). CIFAR numbers are from their own PyTorch re-implementation with tuned baselines.

**holm1979simple**: Scand. J. Statist. 6(2):65-70 (JSTOR 4615733); confirmed through multiple secondary citations, no primary page fetched.
**schuirmann1987comparison**: J. Pharmacokinet. Biopharm. 15(6):657-680, DOI 10.1007/BF01068419 (confirmed via USP <1090> citation and DOI listing; secondary).
**zhang2018mixup**: ICLR 2018 (iclr.cc/virtual/2018/poster/177; arXiv 1710.09412). Authors Zhang, Cisse, Dauphin, Lopez-Paz.
**szegedy2016rethinking**: CVPR 2016, pp 2818-2826 (openaccess.thecvf.com landing page). Authors Szegedy, Vanhoucke, Ioffe, Shlens, Wojna. Label smoothing is in Sec. 7.
**cubuk2020randaugment_neurips**: NeurIPS 2020 vol 33 pp 18613-18624 (proceedings page); CVPRW pp 702-703.
**le2015tiny**: 200 classes, 64x64 images, 500 train / 50 val / 50 test per class (per dataset page); report is a CS231N course report; the original PDF was not located.

## Not fully verified
- le2015tiny: primary report/PDF not found; entry follows commonly cited form (howpublished only, no URL).
- inoue2020semi: APSIPA ASC 2020 venue from secondary sources only; no pages.
- holm1979simple, schuirmann1987comparison: only secondary sources (JSTOR/DOI/USP citation), consistent.
- oliver2018realistic pages (3235-3246) and ICLR papers' pages: not on official pages, omitted.
- zheng2023coverage headline numbers differ across versions (see above).
- DBLP not usable (bot challenge); Semantic Scholar rate-limited.
