(TeX-add-style-hook
 "scale"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("standalone" "tikz")))
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"
    "stix"
    "tikz"
    "amsmath"
    "xfrac"
    "graphicx"
    "bm")
   (TeX-add-symbols
    '("scalebar" ["argument"] 4))))

