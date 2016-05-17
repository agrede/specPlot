(TeX-add-style-hook
 "loglogplot"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("standalone" "tikz")))
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"
    "stix"
    "pgfplots"
    "amsmath"
    "xfrac"
    "bm")))

