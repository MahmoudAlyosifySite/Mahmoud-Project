# Budget-reporting audit: do published label-efficiency comparisons equalise training budget?

> **Third revision (2026-11-26).** On a second reading, Cole et al. (2022) do not meet the inclusion criterion: all
> their pools of different size are used for self-supervised pretraining without labels, and their only
> labeled-subset arm (supervised training from scratch, their Figure 2) is plotted against the number of labeled
> images, not compared with a larger pretraining pool. The paper therefore counts 16 applicable papers: 7 matched or
> accounted, 3 unequal but disclosed, 1 equal epochs over pools of different size, 5 unclear. The audit had a single
> reader; the one paper in the "equal epochs" category, Benato et al. (2023), was read a second time in full
> before the paper names it (below). The tables and counts further down are those of the first reading.
>
> **Second reading of Benato et al. (2023)** (arXiv 2302.02663, version 1, the only version; no published version is
> mentioned; full HTML text read). Verdict confirmed. Arms: "SupCon: Train with A on I_S and L_S" and "SimCLR:
> Train with A on I_{S∪U}" (Sec. 4.4.1); split: "a supervised training-set S containing 1% supervised samples from D,
> an unsupervised training-set U with 69% of the samples in D, and a test set T with 30%" (Sec. 4.2), i.e. 1% against
> 70% of the data (Table 2: e.g. P. cysts S 95 / U 6,602). Budget: "We use 50 epochs and select the best model
> through a checkpoint obtained from the lowest validation loss during training" (Sec. 4.3), and Table 3: "Both
> methods are compared trained from scratch and with pre-trained weights during 50 epochs." No batch size, steps,
> iterations or compute are stated. The paper discusses the size difference in samples ("SimCLR – even trained with
> thousands of unsupervised samples (69%) – ... could not overpass SupCon which used only dozens of supervised samples
> (1%)", Sec. 5.2) and explains the result by data separation, not by training effort. With an equal batch size the
> SimCLR arm would take about 70 times the gradient steps; this ratio is our inference, since the batch size is not
> given.

Method: 26 papers read via ar5iv HTML (text search of setup/appendix sections, then reading hits). Limits: ar5iv strips some math/numbers; FixMatch (conversion failed) and Inoue & Goto 2020 (speaker-recognition, not images) were NOT audited; one reader, no code inspected (so "UNCLEAR" = not recoverable from text). BibTeX: DBLP is bot-walled for automated requests, so metadata is from the arXiv API and venues are given only when arXiv's own fields state them (see budget_audit.bib header).

## Table

| key | arms & pool sizes | budget unit per arm | same #updates? | verdict |
|---|---|---|---|---|
| Assran2020SuNCEt | SimCLR vs SimCLR+SuNCEt; both full ImageNet, +1/10% labels in extra loss | epochs of full set; per-iteration cost matched; compute-to-match curves | yes by design | MATCHED |
| Assran2021PAWS | PAWS (ImageNet + 10% labels) vs SimCLRv2, BYOL, SwAV, FixMatch, MPL | epochs on unlabeled batches, with notes (MPL teacher updates) | stated/accounted, not equal | MATCHED |
| Zhai2019S4L | supervised on 10% labels vs S4L (10% labels + all unlabeled) | epochs over LABELED set only | no (S4L sees more images), disclosed | OTHER-MISMATCH |
| Chen2020SimCLRv2 | SimCLRv2 800 ep full set + FT on 1/10% vs literature supervised | epochs (pretrain full set; FT labeled set); baseline not stated | cannot be determined | UNCLEAR |
| Henaff2019CPCv2 | CPC on full ImageNet (200 ep) + classifier vs supervised on 1-100% subsets | CPC epochs; baseline budget not stated | cannot be determined | UNCLEAR |
| Zoph2020Rethinking | COCO 20-100% labels x {scratch, sup., self-sup. init, self-training} | iterations, each arm trained to convergence | no, deliberately | OTHER-MISMATCH (benign) |
| Li2020CoMatch | CoMatch vs literature table (ImageNet 1/10%) | one "#Epochs" column, definition not given | cannot be determined | UNCLEAR |
| Zheng2022SimMatch | SimMatch vs S4L, FixMatch, PAWS, SimCLR... (ImageNet 1/10%) | epochs + FLOPs-per-epoch remark for PAWS | partially accounted | MATCHED (partial) |
| Yang2022CCSSL | FixMatch/MixMatch +/- CCSSL, same pools | epochs (512 all) | yes, same pool | NOT-APPLICABLE |
| Fini2023Daino | Daino vs DINO/MSN/PAWS/SimMatch/CoMatch (ImageNet 1/10%) | separate epochs (unlab./lab.) and batch sizes columns | not equal, fully reported | MATCHED |
| Islam2021Broad | CE/SupCon/SelfSupCon on full ImageNet | epochs | same pool | NOT-APPLICABLE |
| Newell2020HowUseful | SSL pretrain on 50k..400k unlabeled; labeled counts vary | epochs rescaled to equal iterations | yes | MATCHED |
| Cole2022When | SimCLR on 50k..1M ImageNet subsets, vs supervised | 1000 epochs of the (varying) pool | no (20x steps 50k vs 1M), undiscussed | EPOCH-MISMATCH (unlabeled-pool variant) |
| Yang2020Rethinking | imbalanced learning +/- SSP | epochs | same pool | NOT-APPLICABLE |
| Liu2021SSLRobust | SL vs SSL on 10k-110k down-sampled variants | epochs rescaled to equal iterations (SSL); SL by CV | yes (SSL) | MATCHED |
| Berthelot2019MixMatch | MixMatch vs baselines, CIFAR/SVHN | total budget not in readable text | cannot be determined | UNCLEAR |
| Wang2022USB | ~15 SSL algorithms, same pools | iterations, equal batch sizes | yes across SSL arms | MATCHED |
| Cai2022SemiViT | sup. FT 1/10% labels scratch vs SSL init vs semi-sup FT | labeled-set epochs 100 vs 500 | no, disclosed | OTHER-MISMATCH |
| Kim2021SelfMatch | SelfMatch/FixMatch/MixMatch, same pools | epochs | same pool | NOT-APPLICABLE |
| Ericsson2021HowWell | 13 SSL + sup. ImageNet models, full ImageNet | authors' epochs (200-800), fixed downstream iterations | n/a | NOT-APPLICABLE |
| Tian2020InfoMin | SSL methods on full ImageNet | 200 epochs for all | same pool | NOT-APPLICABLE |
| Khosla2020SupCon | SupCon vs CE, full labeled ImageNet | epochs | same pool | NOT-APPLICABLE |
| Zhang2021CoCalibration | own (200 ep) vs literature | single "epochs" column, undefined | cannot be determined | UNCLEAR |
| Benato2023Linking | SimCLR on all samples vs SupCon on small supervised subset | "50 epochs" both | no (factor = pool ratio, not discussed) | EPOCH-MISMATCH |
| Zheng2021WCL | self-supervised only | epochs | n/a | NOT-APPLICABLE |
| Takaya2025ModAn | multi-label SupCon vs SimCLR vs CE, same pool | 1000 epochs all | same pool | NOT-APPLICABLE |

## Summary counts (26 papers)
MATCHED 7 (SuNCEt, PAWS, SimMatch-partial, Daino, Newell, Liu, USB); EPOCH-MISMATCH 2 (Benato 2023; Cole 2022, unlabeled-pool variant); OTHER-MISMATCH 3 (S4L, Cai, Zoph-benign); UNCLEAR 5 (SimCLRv2, CPCv2, CoMatch, MixMatch, Zhang 2021); NOT-APPLICABLE 9 (listed, not counted). Of the 17 applicable papers: 7 matched/accounted, 2 epoch-mismatch, 3 disclosed-unequal, 5 unreadable.

## Honest conclusion
1. The strict pitfall (equal epochs, SupCon/supervised on a small labeled pool vs SSL on a bigger pool, no step accounting) was found clearly in only one paper (Benato et al. 2023, an application paper), plus one close variant that varies the unlabeled pool size at a fixed 1000 epochs (Cole et al. 2022); both are in the pool-size-vs-epochs form, and Cole is not about labels.
2. The stronger, better-known papers either match iterations explicitly (Newell & Deng, Liu et al.), keep per-iteration cost equal (SuNCEt), or report separate labeled/unlabeled epoch columns (PAWS, Daino, SimMatch), which shows the problem is recognised, not widespread.
3. The more common failure is non-reporting: five of 17 applicable papers state a single undefined "epochs" column or omit the baseline's budget, so the reader cannot check equality (the reviewer's "opposite problem"); this is real but mostly a feature of tables copied from earlier papers.
4. Caveats: 26 papers, text-only reading with some numbers lost in conversion, no code checked, FixMatch not read, and the sample was chosen from a list that favours well-run SSL work rather than low-label SupCon applications, where the trap is likelier.
5. Recommendation: do not present the pitfall as a finding supported by a prevalence audit; keep it as a methodological precaution (cite Newell & Deng, Liu et al., SuNCEt, PAWS as precedent for matching steps; cite Benato 2023 and Cole 2022 as examples of unmatched-epoch reports), and state it as "budgets are often unstated or unit-ambiguous" rather than "commonly mis-matched".


## Evidence paragraphs (batch 1)

**Assran2020SuNCEt** (Assran, Ballas, Castrejon, Rabbat; "Supervision Accelerates Pre-training in Contrastive Semi-Supervised Learning of Visual Representations"; arXiv 2006.10803). Arms: SimCLR (all ImageNet, unlabeled) vs SimCLR+SuNCEt (same images, 1%/10% labeled used in an added loss), plus SimCLR+CE. Budget unit: epochs over the full set, plus explicit compute accounting (Fig. 3, compute-to-match). Same updates: yes, per-iteration cost deliberately matched. Verdict: MATCHED. Quote (App. implementation details): "we aim to keep the cost per-iteration roughly the same as the baseline, so we use a smaller unsupervised batch-size." https://ar5iv.labs.arxiv.org/html/2006.10803

**Assran2021PAWS** (arXiv 2104.13963). Arms: PAWS (full ImageNet unlabeled + 10% labeled) vs SimCLRv2/BYOL/SwAV/FixMatch/MPL. Unit: epochs counted on the unlabeled mini-batches, with explicit caveats. Same updates: not equalised across methods, but unit stated and extra teacher updates of MPL accounted. Verdict: MATCHED (explicitly accounted). Quote (Table 1 footnote): "For label propagation methods, the number of epochs is counted with respect to the unsupervised mini-batches." and "[MPL] does not count the additional 500,000 teacher-network updates". https://ar5iv.labs.arxiv.org/html/2104.13963

**Zhai2019S4L** (arXiv 1905.03670). Arms: supervised baseline on 10% labels vs S4L (10% labels + all unlabeled). Unit: "epochs" defined over the LABELED subset only. Same updates: no, S4L sees many more images per labeled epoch; disclosed. Verdict: OTHER-MISMATCH (disclosed, not equalised). Quote (Sec. 4): "We always define epochs in terms of the available labeled data, i.e. one epoch corresponds to one full pass through the labeled data, regardless of how many unlabeled examples have been seen." https://ar5iv.labs.arxiv.org/html/1905.03670

**Chen2020SimCLRv2** (arXiv 2006.10029). Arms: SimCLRv2 pretrain (full unlabeled ImageNet) + finetune on 1%/10% labels vs supervised baselines taken from the literature. Unit: epochs of the full set for pretraining (800), epochs of labeled subset for fine-tuning (60/30); baseline budget not stated. Same updates: cannot be determined. Verdict: UNCLEAR (no matching claimed). Quote (Sec. 3.1): "total of 800 epochs" and "we fine-tune for 60 epochs with 1% of labels, and 30 epochs with 10% of labels". https://ar5iv.labs.arxiv.org/html/2006.10029

**Henaff2019CPCv2** (arXiv 1905.09272). Arms: supervised ResNets trained separately on 1%-100% subsets vs CPC-pretrained (unlabeled full ImageNet) + classifier. Unit: CPC pretraining "200 epochs" (Appendix); supervised baseline budget not stated; fine-tuning "a small number of epochs (chosen by cross-validation)". Verdict: UNCLEAR. Quote (App. A): "We train the network for the CPC objective using the Adam optimizer ... for 200 epochs". https://ar5iv.labs.arxiv.org/html/1905.09272

**Zoph2020Rethinking** (arXiv 2006.06882). Arms: COCO subsets (20%-100% labels) from random init / supervised / self-supervised ImageNet init / self-training with ImageNet unlabeled. Unit: iterations, each arm trained to convergence on a held-out set. Same updates: no, deliberately not equal. Verdict: OTHER-MISMATCH in the benign sense (budget tuned per arm and reported in iterations; not the trap). Quote (Sec. 3): "we allow each model to train until it converges ... training takes 45k iterations with Augment-S1 and 120k iterations with Augment-S4". https://ar5iv.labs.arxiv.org/html/2006.06882

## Evidence paragraphs (batch 2)

**Li2020CoMatch** (arXiv 2011.11183). Arms: CoMatch (ImageNet, 1%/10% labels + unlabeled) vs literature supervised baseline, S4L, FixMatch, SimCLR, BYOL, SwAV... Unit: one "#Epochs" column mixing methods; epoch definition (labeled vs unlabeled) not given; supervised baseline listed as 20 epochs (copied from S4L, labeled-set epochs). Same updates: cannot be determined. Verdict: UNCLEAR. Quote (Table 2 header): "Method #Epochs #Paramters (train/test) ... Supervised baseline [40] 20 ... FixMatch w. DA 400 ... CoMatch 400 ... SimCLR 1000". Text claims "less training time" but gives no step count. https://ar5iv.labs.arxiv.org/html/2011.11183

**Zheng2022SimMatch** (arXiv 2203.06915). Arms: SimMatch (ImageNet 1%/10%) vs S4L, FixMatch, PAWS, SimCLR... Unit: "Epochs" column, plus an explicit FLOPs adjustment for PAWS. Same updates: partly accounted for (not for the S4L rows, whose epochs are over labeled data). Verdict: MATCHED (partial, explicit discussion). Quote (Sec. 4.2): "the reported 300 epochs PAWS should have similar training FLOPS with 1200 epochs SimMatch." https://ar5iv.labs.arxiv.org/html/2203.06915

**Yang2022CCSSL** (arXiv 2203.02261). All arms (FixMatch, MixMatch, +CCSSL) use the same labeled subset and same unlabeled pool, same epochs ("We train 512 epochs for all experiments"). No small-pool vs full-pool contrast. Verdict: NOT-APPLICABLE (listed, not counted). https://ar5iv.labs.arxiv.org/html/2203.02261

**Fini2023Daino** (arXiv 2306.07483, "Semi-supervised learning made simple with self-supervised clustering", CVPR 2023). Arms: Daino (ImageNet 1%/10% labeled + unlabeled) vs DINO/MSN/SimMatch/PAWS/CoMatch. Unit: Tables 2-3 give separate columns "Epochs Unlab. / Lab." and batch sizes for unlabeled and labeled streams; footnote "epochs on labeled data". Same updates: not equal but transparently reported so a reader can compute images seen. Verdict: MATCHED (explicitly accounted/reported). Quote (Table 3 caption): "refers to epochs on labeled data." and column headers "Epochs Unlab. Lab.; Batch size Unlab. Lab." https://ar5iv.labs.arxiv.org/html/2306.07483

**Islam2021Broad** (arXiv 2103.13517, ICCV 2021). CE, SupCon, SelfSupCon, and combinations all pretrained on the same full ImageNet with matched settings; no labeled-subset vs full-set contrast found in the text. Verdict: NOT-APPLICABLE. https://ar5iv.labs.arxiv.org/html/2103.13517

**Newell2020HowUseful** (arXiv 2003.14323, CVPR 2020). Arms: self-supervised pretraining on varying numbers of unlabeled images vs from scratch with varying labeled counts. Unit: epochs but explicitly rescaled. Same updates: yes by design. Verdict: MATCHED. Quote (Fig. 10 caption): "Because 50k unlabeled images is 1/8th the base 400k, the model is trained for more epochs to match total number of training iterations." https://ar5iv.labs.arxiv.org/html/2003.14323

**Cole2022When** (arXiv 2105.05837, CVPR 2022). Arms: SimCLR pretrained on ImageNet subsets from 1M down to 50k images (unlabeled, no labels) vs supervised, linear eval. Unit: epochs of the (varying) pretraining set, fixed 1000. Same updates: no, 20x fewer steps at 50k vs 1M; no step/iteration discussion found (searched "steps/iterations/same number of/longer"); conclusions are phrased as accuracy vs pretraining set size and "pretraining time". Verdict: EPOCH-MISMATCH (pool-size variation, unlabeled; same mechanism as the pitfall). Quote (App. C): "We train with a batch size of 4096 for 1000 epochs"; and (Sec. 4.1) "we can trade a small decrease in accuracy for a 50% decrease in pretraining time." https://ar5iv.labs.arxiv.org/html/2105.05837

**Yang2020Rethinking** (arXiv 2006.07529, NeurIPS 2020). SSP/semi-supervised on imbalanced labeled data; classifier stage "train all models for 200 epochs on CIFAR-LT" with same-pool arms; no labeled-subset vs full-set pretraining contrast made. Verdict: NOT-APPLICABLE. https://ar5iv.labs.arxiv.org/html/2006.07529

**Liu2021SSLRobust** (arXiv 2110.05025, ICLR 2022). Arms: SL vs SSL on down-sampled / imbalanced dataset variants (10k-110k to full). Unit: epochs, explicitly rescaled for SSL; SL epochs chosen by cross-validation. Same updates: yes for SSL arms by design. Verdict: MATCHED. Quote (App. A): "We train the models for more epochs on the down sampled variants to ensure the same number of total iterations." https://ar5iv.labs.arxiv.org/html/2110.05025

**Berthelot2019MixMatch** (arXiv 1905.02249). Semi-supervised on CIFAR/SVHN; the readable text states only "ramp up ... over the first 16,000 steps of training"; the total budget and the supervised baseline's budget are not stated in the text we could read (inherits Oliver et al. 2018 protocol). Verdict: UNCLEAR. https://ar5iv.org/html/1905.02249

**Sohn2020FixMatch** (arXiv 2001.07685): ar5iv conversion failed and no PDF text tool available in the audit environment, so NOT AUDITED (dropped from counts).

## Evidence paragraphs (batch 3)

**Wang2022USB** (arXiv 2208.07204, NeurIPS D&B 2022). Arms: ~15 SSL algorithms (labeled subset + unlabeled pool, same pool for all) and a supervised-labeled-only reference. Unit: training iterations, one fixed total per setting; labeled and unlabeled batch sizes equal. Same updates across SSL algorithms: yes. Supervised-only baseline budget: not verified in our read. Verdict: MATCHED (for the SSL arms). Quote (Sec. 1): "using pre-trained Vision Transformers (ViT) can largely reduce the number of training iterations (e.g., by 80% from 1,000k to 200k on CV tasks)"; (App. G.1) "The labeled and unlabeled batch size is both set to" [number lost in conversion]. https://ar5iv.labs.arxiv.org/html/2208.07204

**Cai2022SemiViT** (arXiv 2208.05688, "Semi-supervised Vision Transformers at Scale", NeurIPS 2022). Arms: supervised fine-tuning on 1%/10% labels from scratch vs from self-supervised (MAE/MoCo-v3/DINO) checkpoints vs semi-supervised fine-tuning. Unit: epochs over the labeled set, different per arm, pretraining checkpoints taken from others with no budget given. Same updates: no (stated differences). Verdict: OTHER-MISMATCH (disclosed, no step accounting). Quote (Sec. 4): "supervised fine-tuning, the model is trained for 100 (500) epochs ... with (without) self-supervised pre-training." https://ar5iv.labs.arxiv.org/html/2208.05688

**Kim2021SelfMatch** (arXiv 2101.06480). All arms (SelfMatch, FixMatch, MixMatch) use the same labeled subset and unlabeled pool; convergence compared in epochs. Verdict: NOT-APPLICABLE. https://ar5iv.labs.arxiv.org/html/2101.06480

**Ericsson2021HowWell** (arXiv 2011.13377, CVPR 2021). Arms: 13 SSL models and supervised ImageNet models, all on full ImageNet; pretraining epochs "as reported by original authors" (Table 10: 200-800 epochs, differing across SSL models); downstream "fixed number of iterations". No labeled-subset pretraining arm. Verdict: NOT-APPLICABLE (budget heterogeneity across SSL models is disclosed but the labeled-subset vs full contrast is absent). https://ar5iv.labs.arxiv.org/html/2011.13377

**Tian2020InfoMin** (arXiv 2005.10243, NeurIPS 2020). All arms pretrained on the same full ImageNet. Quote (Table 2): "For fair comparison, InstDis, PIRL, MoCo, and InfoMin are all pre-trained for 200 epochs." Verdict: NOT-APPLICABLE. https://ar5iv.labs.arxiv.org/html/2005.10243

**Khosla2020SupCon** (arXiv 2004.11362, NeurIPS 2020). SupCon and cross-entropy on full labeled ImageNet; no labeled-subset arm. Verdict: NOT-APPLICABLE. https://ar5iv.labs.arxiv.org/html/2004.11362

**Zhang2021CoCalibration** ( arXiv 2105.07387, "Semi-supervised Contrastive Learning with Similarity Co-calibration"). Arms: own method (200 epochs) vs literature table with a single "epochs" column (Supervised 20, Pseudo-label 100, Fixmatch 400, S4L 200). Epoch definitions not given; baseline numbers copied from original papers. Verdict: UNCLEAR. Quote (Table 2): "Supervised 20 ... Pseudo-label [19] 100 ... Fixmatch(w. RA) [26] 400 ... S4L-Rotation [35] 200". https://ar5iv.labs.arxiv.org/html/2105.07387

**Benato2023Linking** (arXiv 2302.02663, "Linking data separation, visual separation, and classifier performance using pseudo-labeling by contrastive learning"). Arms: SimCLR on all available samples vs SupCon on the small supervised training subset (and a hybrid), compared in DS/VS tables. Unit: epochs only; Table 3 caption: both compared "trained from scratch and with pre-trained weights during 50 epochs". Same updates: no, SupCon sees only the small labeled set per epoch (factor = pool ratio; exact fraction lost in conversion), not discussed. Verdict: EPOCH-MISMATCH (clearest instance found: small labeled pool for SupCon, larger pool for SimCLR, same epochs, no step count). Quote: "(a) using all samples available ... SimCLR; (b) using our supervised samples with SupCon" (Sec. 3.1) and "compared ... during 50 epochs" (Table 3). https://ar5iv.labs.arxiv.org/html/2302.02663

**Zheng2021WCL** (arXiv 2110.04770, ICCV 2021, "Weakly Supervised Contrastive Learning"). Self-supervised, no labeled-subset arm. Verdict: NOT-APPLICABLE. https://ar5iv.labs.arxiv.org/html/2110.04770

**Takaya2025ModAn** (arXiv 2508.18613, 2025). Arms: multi-label SupCon vs SimCLR vs CE, all on the same pretraining pool (miniRIN), all 1000 epochs (Table 2). Same pool so same steps. Verdict: NOT-APPLICABLE. https://ar5iv.labs.arxiv.org/html/2508.18613

**Inoue2020** (candidate in list): is a speaker-recognition paper, not images; the arXiv id we first tried (2007.11004) was a wrong guess, so it was not audited (dropped).
