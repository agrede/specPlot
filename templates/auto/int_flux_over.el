(TeX-add-style-hook
 "int_flux_over"
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

