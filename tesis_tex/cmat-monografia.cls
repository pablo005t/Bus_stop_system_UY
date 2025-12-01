%% Clase para monografías del CMat
%%
%% Autor: Gonzalo Tornaría <tornaria@cmat.edu.uy>
%%
\ProvidesClass{cmat-monografia}
  [2017/11/23 v0.2 CMat clase de monografía]
%% 2016/11/23 v0.2:
%%  - posición del título
%% 2016/10/31 v0.1:
%%  - ajusta geometría
%% 2016/10/12 v0.0:
%%  - primera version

\NeedsTeXFormat{LaTeX2e}[1995/12/01]

\def\mono@class{amsbook}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\mono@class}}
\ProcessOptions\relax

\PassOptionsToClass{11pt}{\mono@class}

\LoadClass{\mono@class}[2014/09/29]

%%% Page geometry

\RequirePackage{geometry}
\geometry{a4paper}
\geometry{text={147mm,207.9mm}} % 0.7 of A4
\geometry{hcentering}

%%% Title page

\def\author#1{\gdef\@author{#1}}
\def\subtitle#1{\gdef\@subtitle{#1}}
\def\director#1{\gdef\@director{#1}}
\def\codirector#1{\gdef\@codirector{#1}}
\def\inst#1{\gdef\@inst{#1}}
\def\coinst#1{\gdef\@coinst{#1}}
\let\@author\@empty
\let\@subtitle\@empty
\let\@director\@empty
\let\@codirector\@empty
\let\@inst\@empty
\let\@coinst\@empty

%%% Nombres en español
\def\abstractname{Resumen}
\def\contentsname{\'Indice general}
\def\chaptername{Cap\'itulo}

\endinput
