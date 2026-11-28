# Cumulative-link mixed models of the BA listening test (pre-specified in listening_test_protocol.md):
#   rating ~ system + (1 | listener) + (1 | voice) + (1 | sentence), logit link, for Egyptian-ness and naturalness;
# contrasts BA - VoxCPM2, BA - Chatterbox V3, VoxCPM2 - Chatterbox V3 (Wald, on the log-odds scale; Holm correction over
# the six contrasts is applied by listening_analysis.py).
#   Rscript listening_clmm.R ratings.csv out.json
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(ordinal))
d <- read.csv(args[1], stringsAsFactors = FALSE)
d$system <- factor(d$system, levels = c("ba", "voxcpm2", "cb-v3", "namaa"))
pairs <- list(c("ba", "voxcpm2"), c("ba", "cb-v3"), c("voxcpm2", "cb-v3"))
out <- list(n = nrow(d), listeners = length(unique(d$listener)), contrasts = list(), fits = list())
for (scale in c("egy", "nat")) {
  d$y <- factor(d[[scale]], levels = 1:5, ordered = TRUE)
  m <- clmm(y ~ system + (1 | listener) + (1 | voice) + (1 | sentence), data = d, link = "logit", Hess = TRUE)
  b <- coef(m)[paste0("system", levels(d$system)[-1])]
  names(b) <- levels(d$system)[-1]
  V <- vcov(m)[paste0("system", levels(d$system)[-1]), paste0("system", levels(d$system)[-1])]
  dimnames(V) <- list(names(b), names(b))
  beta <- c(ba = 0, b)          # BA is the reference level
  for (p in pairs) {
    a <- p[1]; z <- p[2]
    est <- beta[[a]] - beta[[z]]
    va <- if (a == "ba") 0 else V[a, a]
    vz <- if (z == "ba") 0 else V[z, z]
    cv <- if (a == "ba" || z == "ba") 0 else V[a, z]
    se <- sqrt(va + vz - 2 * cv)
    out$contrasts[[length(out$contrasts) + 1]] <- list(scale = scale, a = a, b = z, est = est, se = se,
      lo = est - 1.96 * se, hi = est + 1.96 * se, p = 2 * pnorm(-abs(est / se)))
  }
  vc <- sapply(VarCorr(m), function(x) as.numeric(attr(x, "stddev")))
  out$fits[[scale]] <- list(logLik = as.numeric(logLik(m)), sd_random = as.list(vc), converged = m$optRes$convergence == 0)
}
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 8), args[2])
