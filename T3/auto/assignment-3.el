(TeX-add-style-hook
 "assignment-3"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("harvardml" "10pt")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "harvardml"
    "harvardml10"
    "url"
    "enumitem"
    "amsfonts"
    "listings"
    "booktabs"
    "graphicx"
    "bm"
    "tikz")
   (TeX-add-symbols
    "given"
    "R"
    "E"
    "var"
    "cov"
    "N"
    "ep")
   (LaTeX-add-environments
    "lemma"))
 :latex)

