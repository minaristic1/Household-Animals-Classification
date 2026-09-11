PREZENTACIJA — Household Animals Classification
================================================

Sadržaj:
  prezentacija.tex   glavni LaTeX fajl (Beamer, tema Szeged + seahorse)
  img/               sve slike, izvučene direktno iz izlaza notebook-a
  prezentacija.pdf   već iskompajlirana verzija

Kompajliranje (obavezno XeLaTeX zbog fontspec/polyglossia):

    xelatex prezentacija.tex
    xelatex prezentacija.tex      # drugi put, zbog sadržaja i navigacije

LOGO FAKULTETA
Naslovni slajd traži fajl "slika1.jpg" (isti kao u prezentaciji iz
Informacionih sistema). Kopiraj ga pored prezentacija.tex i logo će se
pojaviti dole levo. Ako fajl ne postoji, prezentacija se svejedno
kompajlira — samo bez logoa.

VERZIJA BEZ ANIMACIJA (za štampu / podsetnik)
U prvoj liniji zameni:
    \documentclass[8pt,a4paper]{beamer}
sa:
    \documentclass[8pt,a4paper,handout]{beamer}
Time se svi \pause ignorišu i dobija se 21 slajd umesto 63 prelaza.
