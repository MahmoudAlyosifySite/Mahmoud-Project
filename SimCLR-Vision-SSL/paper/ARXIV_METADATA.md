# arXiv submission form: values to paste

Upload `arxiv_submission.tar.gz` (built by `bash make_arxiv_package.sh`); main file `arxiv.tex`.
arXiv requires that every author agrees to the posting.

**Title**

Where Do Scarce Labels Belong? A Compute-Matched Study of Contrastive Pretraining

**Authors**

Mahmoud Alyosify, Mirna Imbabi, Natalie Nashed, Hazem M. Abbas

**Abstract** (1,500 characters; arXiv allows 1,920)

Supervised contrastive learning can use labels during pretraining, but in practice labels are scarce and unlabeled images are plentiful. Should the few labels go inside the contrastive loss, or be kept until after label-free pretraining? Two confounds make this hard to answer: the compute that each method receives, and evaluations that use labels the practitioner does not have. We remove both with a compute-matched 2x2 design that separates the use of labels from the choice of images: SimCLR and supervised contrastive learning (SupCon), each trained on the labeled subset or on all images, with the same number of gradient steps, on CIFAR-10, CIFAR-100, STL-10 and Tiny-ImageNet. Three results stand out. First, labels inside the loss buy speed, not a better end point: with 10% of the CIFAR-10 labels, SupCon leads SimCLR by 13.3 points after 970 steps and trails it by 6.6 points after 97,000. Second, the evaluation protocol decides the ranking: the standard linear probe, trained with all labels, ranks a semi-supervised SupCon first, while fine-tuning on the same 5,000 labels ranks it last. Third, a simple diagnostic explains the reversal: encoders that use labels make classes tight on the labeled images, but not on images seen without a label or never seen. Within contrastive pretraining, scarce labels are better kept out of a SupCon-style loss and spent afterwards: in fine-tuning, in a short supervised contrastive stage or, best in our experiments, in semi-supervised training.

**Comments**

30 pages, 4 figures, 20 tables. Code and per-run results: https://github.com/MahmoudAlyosifySite/Mahmoud-Project/tree/main/SimCLR-Vision-SSL

**Primary category**: cs.LG (Machine Learning). **Cross-list**: cs.CV (Computer Vision and Pattern Recognition).

**License**: the authors' choice. CC BY 4.0 is the most open option and is compatible with Elsevier's preprint
policy ("authors can share their preprint anywhere at any time").

**After the journal submission**: when the article is accepted, add the journal reference and DOI to the arXiv
record ("Journal-ref" and "DOI" fields); do not replace the preprint with the publisher's typeset PDF.

If the abstract or the page count changes, regenerate this file's numbers from `arxiv.pdf`.
