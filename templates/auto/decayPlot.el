(TeX-add-style-hook
 "decayPlot"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("standalone" "tikz")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("siunitx" "use-xspace=true" "redefine-symbols=false" "range-units=single")))
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"
    "stix"
    "pgfplots"
    "amsmath"
    "xfrac"
    "bm"
    "siunitx")
   (LaTeX-add-siunitx-units
    "decade"
    "arb")))

