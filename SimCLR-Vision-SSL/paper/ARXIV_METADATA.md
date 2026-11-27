# arXiv submission form: values to paste

Upload `arxiv_submission.tar.gz` (built by `bash make_arxiv_package.sh`); main file `arxiv.tex`.
arXiv requires that every author agrees to the posting.

**Title**

Where Do Scarce Labels Belong? A Compute-Matched Study of Supervised and Self-Supervised Contrastive Pretraining

**Authors**

Mahmoud Alyosify, Mirna Imbabi, Natalie Nashed, Hazem M. Abbas

**Abstract** (1,626 characters; arXiv allows 1,920)

Supervised contrastive learning can use labels during pretraining, but in practice labels are scarce and unlabeled images are plentiful. Should the few labels go inside the contrastive loss, or be kept until after label-free pretraining? Two confounds hide the answer: unequal compute, and evaluations that use labels the practitioner does not have. We remove both with a compute-matched 2x2 design: SimCLR and supervised contrastive learning (SupCon), each trained on the labeled subset or on all images for the same number of gradient steps, on CIFAR-10, CIFAR-100, STL-10 and Tiny-ImageNet. Three results stand out. First, labels inside the loss buy speed, not a better end point: with 10% of the CIFAR-10 labels, SupCon leads SimCLR by 13.3 points after 970 steps and trails it by 6.6 points after 97,000. Second, the standard linear probe, trained with all labels, overstates label-using encoders: after 9,700 steps it puts a semi-supervised SupCon 1.9 points ahead of SimCLR, whereas fine-tuning on the same 5,000 labels puts it 3.9 points behind, or level when each encoder gets its best recipe; with longer training SimCLR leads under every probe and recipe. Third, a seen/unseen diagnostic agrees: label-using encoders make classes several times tighter on their labeled images than on any other images. Within contrastive pretraining, scarce labels are better spent after label-free pretraining: in fine-tuning, in a short supervised contrastive stage for a frozen encoder or, best in our experiments, in semi-supervised training. Budgets should be counted in steps, and encoders evaluated with the available labels.

**Comments**

35 pages, 4 figures, 24 tables. Code and per-run results: https://github.com/MahmoudAlyosify1/scarce-labels-contrastive

**Primary category**: cs.LG (Machine Learning). **Cross-list**: cs.CV (Computer Vision and Pattern Recognition).

**License**: the authors' choice. CC BY 4.0 is the most open option and is compatible with Elsevier's preprint
policy ("authors can share their preprint anywhere at any time").

**After the journal submission**: when the article is accepted, add the journal reference and DOI to the arXiv
record ("Journal-ref" and "DOI" fields); do not replace the preprint with the publisher's typeset PDF.

If the abstract or the page count changes, regenerate this file's numbers from `arxiv.pdf`.
