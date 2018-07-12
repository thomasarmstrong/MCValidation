%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% GCT/CTA Template  Version 1.0 -- Edited for GCT, based on cta.cls version 1.2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% 1.0 Inital release
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{gct}

\RequirePackage{etoolbox}

%% Options for short and long documents
\newbool{docstyle}		%% this boolean is set true for long and false for short documents
	\DeclareOption{long}{\booltrue{docstyle}}
	\DeclareOption{short}{\boolfalse{docstyle}}

%% options for footer
\newcounter{foot}
	\DeclareOption{CTAP}{\setcounter{foot}{0}}
	\DeclareOption{CTAO}{\setcounter{foot}{1}}
	\DeclareOption{CTAC}{\setcounter{foot}{2}}
  	\DeclareOption{GCT}{\setcounter{foot}{3}}

\ExecuteOptions{GCT,short}		%% tells which is the default option
\ProcessOptions\relax		%% Process given options

%\ifnum\value{foot}=0 \LoadClass{report}
\ifbool{docstyle}{\LoadClass{book}}{\LoadClass{report}}
\RequirePackage[figuresright]{rotating}
\RequirePackage{color}
\RequirePackage{graphicx}
\RequirePackage{multicol}
\RequirePackage{setspace}
\RequirePackage{colortbl}
\RequirePackage{longtable}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage{tocloft}
\RequirePackage[colorlinks=true,pdffitwindow=true, pdfstartview=FitH]{hyperref}
\RequirePackage{helvet}
\RequirePackage[margin=2.5cm, a4paper]{geometry}
\RequirePackage{calc}
\RequirePackage{datetime}
\RequirePackage{enumitem}
\RequirePackage{tikz}
\RequirePackage{placeins}
\RequirePackage[toc,nonumberlist]{glossaries}
\RequirePackage[titletoc]{appendix}
\RequirePackage{hhline}
\RequirePackage{nicefrac}
\RequirePackage{amssymb}
\RequirePackage{float}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Definition of date format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newdateformat{ctadate}{\twodigit{\THEDAY}\ \monthname[\THEMONTH] \THEYEAR}
\newdateformat{tabledate}{\THEYEAR-\twodigit{\THEMONTH}-\twodigit{\THEDAY}}
\setdefaultdate{\ctadate}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Font & Appendix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\familydefault}{\sfdefault}
\renewcommand{\appendixname}{Appendix}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Colors and hyperlinks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\definecolor{ctagray}{RGB}{200,200,200}
\definecolor{ctadarkgray}{RGB}{25,25,25}
\definecolor{ctablue}{RGB}{0,30,82}
\definecolor{ctalightblue}{RGB}{178,188,203}     % 70% transparency of ctablue
\definecolor{tablightblue}{RGB}{220,230,241}
\definecolor{tabmedblue}{RGB}{184,204,228}
\definecolor{tabinterblue}{RGB}{149,179,215}
\definecolor{tabdarkblue}{RGB}{83,141,213}

\color{ctadarkgray}

\newcommand\ctadarkgray[1]{\textcolor{ctadarkgray}{#1}}
\newcommand\ctagray[1]{\textcolor{ctagray}{#1}}
\newcommand\ctablue[1]{\textcolor{ctablue}{#1}}
\newcommand\ctablueUP[1]{\textcolor{ctablue}{\MakeUppercase{#1}}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Custom commands
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\global\let\@version\@empty
\global\let\@history\@empty
\global\let\@acronym\@empty
\global\let\@addref\@empty
\global\let\@abbreviations\@empty
\global\let\@keywords\@empty
\global\let\@docs\@empty
\global\let\@approvedby\@empty
\global\let\@distribution\@empty
\global\let\@realtitle\@title
\global\let\@realdate\@date

\global\let\@editor\@empty
\global\let\@approver\@empty
\newcommand{\editor}[1]{\def\@editor{#1}}
\newcommand{\approver}[1]{\def\@approver{#1}}

\newcommand{\version}[1]{\def\@version{#1}}
\newcommand{\history}[1]{\def\@history{#1}}
\newcommand{\acronym}[1]{\def\@acronym{#1}}
\newcommand{\addref}[1]{\def\@addref{#1}}
\newcommand{\keywords}[1]{\def\@keywords{#1}}
\newcommand{\abbreviations}[1]{\def\@abbreviations{#1}}
\newcommand{\docs}[1]{\def\@docs{#1}}
\newcommand{\approvedby}[1]{\def\@approvedby{#1}}
\newcommand{\distribution}[1]{\def\@distribution{#1}}


\global\let\@project\@empty
\newcommand{\project}[1]{\def\@project{#1}}

\global\let\@shorttitle\@empty
\newcommand{\shorttitle}[1]{\def\@shorttitle{#1}}

\global\let\@checkedby\@empty
\newcommand{\checkedby}[1]{\def\@checkedby{#1}}

\global\let\@comment\@empty
\newcommand{\comment}[1]{\def\@comment{#1}}



\renewcommand{\bibname}{References}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Load Packages with customized commands
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[margin=10pt,font={footnotesize, color=ctablue},labelfont=bf,labelsep=endash]{caption}

\setglossarystyle{super3col}
\setlength{\glsdescwidth}{\linewidth}
\makeglossaries


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% The CTA logo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\ctalogo}{
\begin{tikzpicture}[y=0.80pt,x=0.80pt,yscale=-1, inner sep=0pt, outer sep=0pt]
\begin{scope}
    \path[fill=ctablue,nonzero rule] (484.6045,681.2852) .. controls
      (295.3795,717.4965) and (123.3932,662.4615) .. (100.4632,558.3602) .. controls
      (88.3682,503.4440) and (120.1845,445.8477) .. (180.7157,398.3627) .. controls
      (99.6445,452.2015) and (53.7732,524.0090) .. (67.4782,591.7052) .. controls
      (89.4845,700.4077) and (256.7507,757.3390) .. (441.0770,718.8627) .. controls
      (527.9795,700.7240) and (603.1495,664.6115) .. (656.0382,620.1190) .. controls
      (608.0895,647.2915) and (549.5532,668.8565) .. (484.6045,681.2852);
    \path[draw=ctablue,line join=round,line cap=round,miter limit=4.00,line
      width=2.000pt] (484.6045,681.2852) .. controls (295.3795,717.4965) and
      (123.3932,662.4615) .. (100.4632,558.3602) .. controls (88.3682,503.4440) and
      (120.1845,445.8477) .. (180.7157,398.3627) .. controls (99.6445,452.2015) and
      (53.7732,524.0090) .. (67.4782,591.7052) .. controls (89.4845,700.4077) and
      (256.7507,757.3390) .. (441.0770,718.8627) .. controls (527.9795,700.7240) and
      (603.1495,664.6115) .. (656.0382,620.1190) .. controls (608.0895,647.2915) and
      (549.5532,668.8565) .. (484.6045,681.2852) -- cycle;
    \path[fill=ctablue,nonzero rule] (307.4420,490.2110) .. controls
      (309.9345,502.0872) and (310.9770,506.4010) .. (314.8958,517.7635) --
      (315.1045,518.2135) .. controls (290.1708,534.7397) and (265.5833,533.1747) ..
      (265.0983,533.1747) .. controls (226.9683,533.1259) and (202.0620,506.7260) ..
      (202.0620,457.8297) .. controls (202.0620,411.4022) and (229.5670,380.1935) ..
      (271.0808,380.1935) .. controls (278.6433,380.1935) and (285.6170,381.8835) ..
      (290.0970,384.8535) -- (290.0808,415.0710) .. controls (283.8895,412.4522) and
      (278.6033,411.8660) .. (272.7883,412.0172) .. controls (253.4233,412.5210) and
      (242.0383,430.4672) .. (242.0383,456.5997) .. controls (242.0383,483.6622) and
      (253.4170,500.3885) .. (272.1733,500.2660) .. controls (281.1670,500.2085) and
      (291.2045,497.6160) .. (307.4420,490.2110);
    \path[fill=ctablue,nonzero rule] (357.9658,342.0268) -- (357.9658,385.7618) --
      (403.3408,385.8130) -- (403.3367,414.3955) -- (357.9655,414.3630) --
      (357.9655,478.2505) .. controls (357.9655,495.7768) and (365.3443,500.0830) ..
      (373.0355,500.0830) .. controls (376.0292,500.0830) and (380.0242,498.2205) ..
      (384.5480,496.5568) .. controls (384.5480,496.5568) and (387.9680,508.9543) ..
      (394.3068,523.4605) .. controls (379.5368,531.2630) and (369.9592,533.2943) ..
      (359.5030,533.2943) .. controls (347.8180,533.2943) and (337.6743,530.5255) ..
      (330.2905,523.7630) .. controls (322.2968,515.7643) and (317.6855,504.3905) ..
      (317.6855,480.7105) -- (317.6855,414.3630) -- (299.8480,414.3630) --
      (299.8480,385.7618) -- (317.6855,385.7618) -- (317.6855,354.6355) .. controls
      (317.6855,354.6355) and (334.6168,346.5868) .. (357.9655,342.0268);
    \path[fill=ctablue,nonzero rule] (464.0496,468.4636) .. controls
      (444.9284,468.6073) and (433.7359,471.9973) .. (433.7359,484.2873) .. controls
      (433.7359,499.0461) and (435.2909,504.2661) .. (446.5209,504.2661) .. controls
      (457.6584,504.2661) and (463.7296,494.9573) .. (463.8959,491.2861) .. controls
      (463.9271,490.5886) and (464.0059,487.2611) .. (464.0496,484.2773) --
      (464.0496,468.4636) -- cycle(503.7159,496.8823) .. controls
      (503.7159,508.8798) and (508.3446,520.4261) .. (510.8671,532.5786) --
      (474.8921,532.5786) .. controls (474.8921,532.5786) and (470.8109,527.9811) ..
      (466.8971,520.2936) -- (466.8158,520.3123) .. controls (458.8246,531.6948) and
      (446.8283,533.7898) .. (433.6071,533.7898) .. controls (410.8496,533.7898) and
      (394.0321,514.7173) .. (393.9383,488.2773) .. controls (393.7771,442.8060) and
      (421.1846,442.5010) .. (463.2071,442.8060) -- (462.8196,428.9285) .. controls
      (462.2946,404.1335) and (440.4658,406.8423) .. (412.7233,413.3323) --
      (412.7683,386.9635) .. controls (421.3746,382.3498) and (429.3008,378.4935) ..
      (450.5196,378.4935) .. controls (496.3383,378.4935) and (503.7158,410.4785) ..
      (503.7158,441.2260) -- (503.7158,496.8823) -- cycle;
    \path[fill=ctablue,nonzero rule] (224.9238,591.9286) .. controls
      (223.9038,592.5286) and (222.1638,593.0086) .. (220.1238,593.0086) .. controls
      (213.7625,593.0086) and (209.5025,587.9086) .. (209.5025,578.3073) .. controls
      (209.5025,569.9673) and (213.7025,563.1861) .. (220.9638,563.1861) .. controls
      (222.5238,563.1861) and (224.2038,563.6061) .. (225.0438,564.0861) --
      (224.1438,567.8661) .. controls (223.5438,567.5661) and (222.3438,567.0861) ..
      (220.8425,567.0861) .. controls (216.1625,567.0861) and (214.1225,572.6073) ..
      (214.1225,578.3073) .. controls (214.1225,585.0873) and (216.7025,589.1686) ..
      (220.9638,589.1686) .. controls (222.1637,589.1686) and (223.2438,588.9286) ..
      (224.2638,588.3886) -- (224.9238,591.9286) -- cycle;
    \path[fill=ctablue,nonzero rule] (229.1217,551.1253) -- (233.7417,551.1253) --
      (233.7417,566.7866) -- (233.8618,566.7866) .. controls (234.4617,565.7666) and
      (235.3618,564.8666) .. (236.3218,564.2066) .. controls (237.3430,563.5466) and
      (238.5430,563.1866) .. (240.1630,563.1866) .. controls (243.8230,563.1866) and
      (246.9430,566.0666) .. (246.9430,573.3878) -- (246.9430,592.6491) --
      (242.3230,592.6491) -- (242.3230,573.9266) .. controls (242.3230,570.3866) and
      (241.5430,567.0866) .. (238.4218,567.0866) .. controls (236.2618,567.0866) and
      (234.6430,568.6466) .. (233.9218,571.7066) .. controls (233.8017,572.3066) and
      (233.7417,573.1466) .. (233.7417,573.8666) -- (233.7417,592.6491) --
      (229.1217,592.6491) -- (229.1217,551.1253) -- cycle;
    \path[fill=ctablue,nonzero rule] (266.0224,575.1273) .. controls
      (266.0824,569.3673) and (263.9224,566.7861) .. (261.7024,566.7861) .. controls
      (258.6424,566.7861) and (257.1424,571.1673) .. (256.9611,575.1273) --
      (266.0224,575.1273) -- cycle(256.9024,578.5473) .. controls
      (256.9024,587.4886) and (260.4424,589.3473) .. (263.7424,589.3473) .. controls
      (265.7836,589.3473) and (267.4024,588.8686) .. (268.5436,588.2686) --
      (269.3236,591.6286) .. controls (267.7024,592.4698) and (265.2424,593.0086) ..
      (262.9024,593.0086) .. controls (255.9424,593.0086) and (252.3411,587.3086) ..
      (252.3411,578.4273) .. controls (252.3411,569.0661) and (256.3611,563.1861) ..
      (262.1824,563.1861) .. controls (268.0624,563.1861) and (270.4636,569.4261) ..
      (270.4636,575.9073) .. controls (270.4636,577.0473) and (270.4036,577.8273) ..
      (270.3424,578.5473) -- (256.9024,578.5473) -- cycle;
    \path[fill=ctablue,nonzero rule] (275.6207,571.1070) .. controls
      (275.6207,568.6470) and (275.4407,565.7670) .. (275.3820,563.6070) --
      (279.5208,563.6070) -- (279.7620,568.2270) -- (279.8807,568.2270) .. controls
      (280.8420,565.4657) and (283.3020,563.1870) .. (286.0020,563.1870) .. controls
      (286.3620,563.1870) and (286.6620,563.1870) .. (286.9620,563.2458) --
      (286.9620,567.9270) -- (286.0020,567.9270) .. controls (283.0620,567.9270) and
      (280.9020,570.6270) .. (280.4220,573.9270) .. controls (280.3007,574.7070) and
      (280.2407,575.6070) .. (280.2407,576.5670) -- (280.2407,592.6483) --
      (275.6207,592.6483) -- (275.6207,571.1070) -- cycle;
    \path[fill=ctablue,nonzero rule] (302.3211,575.1273) .. controls
      (302.3811,569.3673) and (300.2199,566.7861) .. (297.9999,566.7861) .. controls
      (294.9399,566.7861) and (293.4399,571.1673) .. (293.2599,575.1273) --
      (302.3211,575.1273) -- cycle(293.1999,578.5473) .. controls
      (293.1999,587.4886) and (296.7399,589.3473) .. (300.0411,589.3473) .. controls
      (302.0811,589.3473) and (303.7011,588.8686) .. (304.8411,588.2686) --
      (305.6211,591.6286) .. controls (304.0011,592.4698) and (301.5411,593.0086) ..
      (299.2011,593.0086) .. controls (292.2399,593.0086) and (288.6399,587.3086) ..
      (288.6399,578.4273) .. controls (288.6399,569.0661) and (292.6599,563.1861) ..
      (298.4811,563.1861) .. controls (304.3611,563.1861) and (306.7611,569.4261) ..
      (306.7611,575.9073) .. controls (306.7611,577.0473) and (306.7011,577.8273) ..
      (306.6411,578.5473) -- (293.1999,578.5473) -- cycle;
    \path[fill=ctablue,nonzero rule] (311.9794,571.1070) .. controls
      (311.9794,567.6870) and (311.7994,565.9470) .. (311.7406,563.6070) --
      (315.7594,563.6070) -- (316.0006,567.1470) -- (316.1206,567.1470) .. controls
      (317.3794,564.8057) and (319.7206,563.1870) .. (322.7819,563.1870) .. controls
      (326.8006,563.1870) and (329.8019,566.1270) .. (329.8019,573.1470) --
      (329.8019,592.6482) -- (325.1819,592.6482) -- (325.1819,573.9870) .. controls
      (325.1819,570.5070) and (324.5206,567.0870) .. (321.1606,567.0870) .. controls
      (319.2406,567.0870) and (317.3794,568.7070) .. (316.8406,571.7670) .. controls
      (316.6606,572.4870) and (316.5994,573.3870) .. (316.5994,574.4070) --
      (316.5994,592.6482) -- (311.9794,592.6482) -- (311.9794,571.1070) -- cycle;
    \path[fill=ctablue,nonzero rule] (341.0211,577.1073) -- (341.1411,577.1073) ..
      controls (341.6811,575.9673) and (342.1611,574.8273) .. (342.7024,573.8673) --
      (348.4624,563.6061) -- (353.4424,563.6061) -- (345.7611,575.8473) --
      (354.1624,592.6486) -- (349.0024,592.6486) -- (342.7024,578.9673) --
      (341.0211,581.6673) -- (341.0211,592.6486) -- (336.4611,592.6486) --
      (336.4611,551.1248) -- (341.0211,551.1248) -- (341.0211,577.1073) -- cycle;
    \path[fill=ctablue,nonzero rule] (360.4596,578.1272) .. controls
      (360.4596,582.9285) and (361.2996,589.4685) .. (365.4996,589.4685) .. controls
      (369.4609,589.4685) and (370.3009,582.6872) .. (370.3009,578.1272) .. controls
      (370.3009,573.6872) and (369.4609,566.8472) .. (365.3809,566.8472) .. controls
      (361.2396,566.8472) and (360.4596,573.6872) ..
      (360.4596,578.1272)(374.9209,578.1272) .. controls (374.9209,589.3485) and
      (369.7596,593.1285) .. (365.2609,593.1285) .. controls (360.0396,593.1285) and
      (355.8396,588.3885) .. (355.8396,578.1872) .. controls (355.8396,567.6272) and
      (360.5809,563.1860) .. (365.5596,563.1860) .. controls (370.7809,563.1860) and
      (374.9209,568.0472) .. (374.9209,578.1272);
    \path[fill=ctablue,nonzero rule] (382.2419,563.6063) -- (385.9019,579.9276) ..
      controls (386.3219,581.9076) and (386.8006,584.4888) .. (387.1619,587.0076) --
      (387.3419,587.0076) .. controls (387.6419,584.6076) and (388.1219,582.1476) ..
      (388.5419,579.9276) -- (392.1419,563.6063) -- (396.7019,563.6063) --
      (389.0819,592.6488) -- (385.0019,592.6488) -- (377.4406,563.6063) --
      (382.2419,563.6063) -- cycle;
    \path[fill=ctablue,nonzero rule] (421.8390,556.8253) -- (421.8390,563.6066) --
      (427.4203,563.6066) -- (427.4203,567.0866) -- (421.8390,567.0866) --
      (421.8390,584.3678) .. controls (421.8390,588.1478) and (423.1603,589.1691) ..
      (424.8390,589.1691) .. controls (425.6190,589.1691) and (426.2203,589.1091) ..
      (426.7002,588.9878) -- (426.9390,592.5291) .. controls (426.0990,592.8278) and
      (425.0190,593.0091) .. (423.5202,593.0091) .. controls (421.6590,593.0091) and
      (420.0390,592.5291) .. (418.9590,591.2678) .. controls (417.7577,589.9491) and
      (417.2190,587.9691) .. (417.2190,583.9478) -- (417.2190,567.0866) --
      (413.6790,567.0866) -- (413.6790,563.6066) -- (417.2190,563.6066) --
      (417.2190,558.6853) -- (421.8390,556.8253) -- cycle;
    \path[fill=ctablue,nonzero rule] (443.6200,575.1273) .. controls
      (443.6800,569.3673) and (441.5188,566.7861) .. (439.2987,566.7861) .. controls
      (436.2387,566.7861) and (434.7387,571.1673) .. (434.5587,575.1273) --
      (443.6200,575.1273) -- cycle(434.4988,578.5473) .. controls
      (434.4988,587.4886) and (438.0388,589.3473) .. (441.3400,589.3473) .. controls
      (443.3800,589.3473) and (445.0000,588.8686) .. (446.1400,588.2686) --
      (446.9200,591.6286) .. controls (445.3000,592.4698) and (442.8400,593.0086) ..
      (440.5000,593.0086) .. controls (433.5387,593.0086) and (429.9388,587.3086) ..
      (429.9388,578.4273) .. controls (429.9388,569.0661) and (433.9588,563.1861) ..
      (439.7800,563.1861) .. controls (445.6600,563.1861) and (448.0600,569.4261) ..
      (448.0600,575.9073) .. controls (448.0600,577.0473) and (448.0000,577.8273) ..
      (447.9400,578.5473) -- (434.4987,578.5473) -- cycle;
    \path[fill=ctablue,nonzero rule] (453.2788,551.1258) -- (457.8988,551.1258) --
      (457.8988,592.6496) -- (453.2788,592.6496) -- (453.2788,551.1258) -- cycle;
    \path[fill=ctablue,nonzero rule] (477.3389,575.1273) .. controls
      (477.4001,569.3673) and (475.2389,566.7861) .. (473.0176,566.7861) .. controls
      (469.9589,566.7861) and (468.4589,571.1673) .. (468.2789,575.1273) --
      (477.3389,575.1273) -- cycle(468.2176,578.5473) .. controls
      (468.2176,587.4886) and (471.7576,589.3473) .. (475.0589,589.3473) .. controls
      (477.1001,589.3473) and (478.7189,588.8686) .. (479.8589,588.2686) --
      (480.6401,591.6286) .. controls (479.0189,592.4698) and (476.5601,593.0086) ..
      (474.2189,593.0086) .. controls (467.2589,593.0086) and (463.6576,587.3086) ..
      (463.6576,578.4273) .. controls (463.6576,569.0661) and (467.6776,563.1861) ..
      (473.4989,563.1861) .. controls (479.3801,563.1861) and (481.7801,569.4261) ..
      (481.7801,575.9073) .. controls (481.7801,577.0473) and (481.7189,577.8273) ..
      (481.6601,578.5473) -- (468.2176,578.5473) -- cycle;
    \path[fill=ctablue,nonzero rule] (486.2183,587.8490) .. controls
      (487.2382,588.5077) and (489.0383,589.3477) .. (490.8983,589.3477) .. controls
      (493.4795,589.3477) and (495.1583,587.7277) .. (495.1583,585.1477) .. controls
      (495.1583,582.9277) and (494.3795,581.4277) .. (491.4382,579.4477) .. controls
      (487.6583,577.0477) and (485.6182,574.5277) .. (485.6182,571.2265) .. controls
      (485.6182,566.4865) and (489.0995,563.1865) .. (493.5983,563.1865) .. controls
      (495.8795,563.1865) and (497.5582,563.9665) .. (498.7595,564.6865) --
      (497.5582,568.1665) .. controls (496.4183,567.3865) and (495.2195,566.8465) ..
      (493.6595,566.8465) .. controls (491.1982,566.8465) and (489.9395,568.6465) ..
      (489.9395,570.5665) .. controls (489.9395,572.4865) and (490.6595,573.5677) ..
      (493.5370,575.5477) .. controls (496.8995,577.7077) and (499.5383,580.3477) ..
      (499.5383,584.3677) .. controls (499.5383,590.1890) and (495.5783,593.0090) ..
      (490.7783,593.0090) .. controls (488.6795,593.0090) and (486.3383,592.3490) ..
      (485.0783,591.3290) -- (486.2183,587.8490) -- cycle;
    \path[fill=ctablue,nonzero rule] (518.4985,591.9286) .. controls
      (517.4785,592.5286) and (515.7373,593.0086) .. (513.6985,593.0086) .. controls
      (507.3360,593.0086) and (503.0760,587.9086) .. (503.0760,578.3073) .. controls
      (503.0760,569.9673) and (507.2773,563.1861) .. (514.5385,563.1861) .. controls
      (516.0985,563.1861) and (517.7785,563.6061) .. (518.6185,564.0861) --
      (517.7173,567.8661) .. controls (517.1185,567.5661) and (515.9185,567.0861) ..
      (514.4185,567.0861) .. controls (509.7385,567.0861) and (507.6973,572.6073) ..
      (507.6973,578.3073) .. controls (507.6973,585.0873) and (510.2785,589.1686) ..
      (514.5385,589.1686) .. controls (515.7373,589.1686) and (516.8185,588.9286) ..
      (517.8398,588.3886) -- (518.4985,591.9286) -- cycle;
    \path[fill=ctablue,nonzero rule] (525.4565,578.1272) .. controls
      (525.4565,582.9285) and (526.2965,589.4685) .. (530.4977,589.4685) .. controls
      (534.4578,589.4685) and (535.2977,582.6872) .. (535.2977,578.1272) .. controls
      (535.2977,573.6872) and (534.4577,566.8472) .. (530.3790,566.8472) .. controls
      (526.2377,566.8472) and (525.4565,573.6872) ..
      (525.4565,578.1272)(539.9190,578.1272) .. controls (539.9190,589.3485) and
      (534.7577,593.1285) .. (530.2590,593.1285) .. controls (525.0365,593.1285) and
      (520.8377,588.3885) .. (520.8377,578.1872) .. controls (520.8377,567.6272) and
      (525.5790,563.1860) .. (530.5565,563.1860) .. controls (535.7790,563.1860) and
      (539.9190,568.0472) .. (539.9190,578.1272);
    \path[fill=ctablue,nonzero rule] (549.5776,582.0275) .. controls
      (549.5776,582.8687) and (549.6364,583.7087) .. (549.8164,584.4287) .. controls
      (550.7176,588.5675) and (552.8176,589.2887) .. (554.0776,589.2887) .. controls
      (557.9176,589.2887) and (559.1776,584.0087) .. (559.1776,578.1875) .. controls
      (559.1776,572.1862) and (557.7364,567.1462) .. (554.0189,567.1462) .. controls
      (551.9164,567.1462) and (549.9976,569.7275) .. (549.7589,572.3662) .. controls
      (549.6364,573.0275) and (549.5776,573.8075) .. (549.5776,574.5875) --
      (549.5776,582.0275) -- cycle(545.0176,572.6675) .. controls
      (545.0176,568.2262) and (544.8364,565.6462) .. (544.7776,563.6062) --
      (548.8576,563.6062) -- (549.0964,567.3262) -- (549.2164,567.3262) .. controls
      (550.8376,564.2662) and (553.1176,563.1862) .. (555.3976,563.1862) .. controls
      (560.4976,563.1862) and (563.7989,568.5262) .. (563.7989,578.0675) .. controls
      (563.7989,588.8687) and (559.3576,593.1287) .. (554.7976,593.1287) .. controls
      (552.0389,593.1287) and (550.4789,591.3887) .. (549.6976,589.7675) --
      (549.5776,589.7675) -- (549.5776,604.1700) -- (545.0176,604.1700) --
      (545.0176,572.6675) -- cycle;
    \path[fill=ctablue,nonzero rule] (581.3770,575.1273) .. controls
      (581.4383,569.3673) and (579.2770,566.7861) .. (577.0557,566.7861) .. controls
      (573.9970,566.7861) and (572.4970,571.1673) .. (572.3170,575.1273) --
      (581.3770,575.1273) -- cycle(572.2557,578.5473) .. controls
      (572.2557,587.4886) and (575.7957,589.3473) .. (579.0970,589.3473) .. controls
      (581.1382,589.3473) and (582.7570,588.8686) .. (583.8970,588.2686) --
      (584.6782,591.6286) .. controls (583.0570,592.4698) and (580.5982,593.0086) ..
      (578.2570,593.0086) .. controls (571.2970,593.0086) and (567.6957,587.3086) ..
      (567.6957,578.4273) .. controls (567.6957,569.0661) and (571.7157,563.1861) ..
      (577.5370,563.1861) .. controls (583.4182,563.1861) and (585.8182,569.4261) ..
      (585.8182,575.9073) .. controls (585.8182,577.0473) and (585.7569,577.8273) ..
      (585.6982,578.5473) -- (572.2557,578.5473) -- cycle;
    \path[fill=ctablue,nonzero rule] (617.0776,577.5877) .. controls
      (614.6751,577.5877) and (609.0964,578.0677) .. (609.0964,584.3677) .. controls
      (609.0964,588.2077) and (611.0151,589.5290) .. (612.6364,589.5290) .. controls
      (614.6751,589.5290) and (616.3576,588.0290) .. (616.8964,585.1477) .. controls
      (617.0164,584.6077) and (617.0776,584.0090) .. (617.0776,583.4677) --
      (617.0776,577.5877) -- cycle(621.5776,586.0490) .. controls
      (621.5776,588.2690) and (621.6351,590.7890) .. (621.8776,592.6490) --
      (617.7364,592.6490) -- (617.4364,589.5890) -- (617.2564,589.5890) .. controls
      (615.9964,591.8090) and (613.9551,593.1290) .. (611.4351,593.1290) .. controls
      (607.5951,593.1290) and (604.5951,589.8890) .. (604.5951,584.8490) .. controls
      (604.5951,577.5277) and (610.4151,574.3477) .. (616.9551,574.2865) --
      (616.9551,573.3877) .. controls (616.9551,569.4865) and (615.9376,566.8465) ..
      (612.3364,566.8465) .. controls (610.5376,566.8465) and (608.9164,567.3865) ..
      (607.5364,568.3465) -- (606.5164,565.1065) .. controls (607.7764,564.2065) and
      (610.4151,563.1865) .. (613.2964,563.1865) .. controls (619.1164,563.1865) and
      (621.5776,566.9665) .. (621.5776,573.4465) -- (621.5776,586.0490) -- cycle;
    \path[fill=ctablue,nonzero rule] (628.2959,571.1070) .. controls
      (628.2959,568.6470) and (628.1146,565.7670) .. (628.0571,563.6070) --
      (632.1946,563.6070) -- (632.4371,568.2270) -- (632.5559,568.2270) .. controls
      (633.5159,565.4657) and (635.9771,563.1870) .. (638.6771,563.1870) .. controls
      (639.0359,563.1870) and (639.3359,563.1870) .. (639.6359,563.2458) --
      (639.6359,567.9270) -- (638.6771,567.9270) .. controls (635.7371,567.9270) and
      (633.5771,570.6270) .. (633.0959,573.9270) .. controls (632.9759,574.7070) and
      (632.9146,575.6070) .. (632.9146,576.5670) -- (632.9146,592.6483) --
      (628.2959,592.6483) -- (628.2959,571.1070) -- cycle;
    \path[fill=ctablue,nonzero rule] (643.4131,571.1070) .. controls
      (643.4131,568.6470) and (643.2344,565.7670) .. (643.1744,563.6070) --
      (647.3144,563.6070) -- (647.5544,568.2270) -- (647.6731,568.2270) .. controls
      (648.6356,565.4657) and (651.0944,563.1870) .. (653.7944,563.1870) .. controls
      (654.1556,563.1870) and (654.4556,563.1870) .. (654.7556,563.2458) --
      (654.7556,567.9270) -- (653.7944,567.9270) .. controls (650.8544,567.9270) and
      (648.6944,570.6270) .. (648.2156,573.9270) .. controls (648.0931,574.7070) and
      (648.0344,575.6070) .. (648.0344,576.5670) -- (648.0344,592.6483) --
      (643.4131,592.6483) -- (643.4131,571.1070) -- cycle;
    \path[fill=ctablue,nonzero rule] (668.7940,577.5877) .. controls
      (666.3940,577.5877) and (660.8127,578.0677) .. (660.8127,584.3677) .. controls
      (660.8127,588.2077) and (662.7340,589.5290) .. (664.3527,589.5290) .. controls
      (666.3940,589.5290) and (668.0740,588.0290) .. (668.6127,585.1477) .. controls
      (668.7327,584.6077) and (668.7940,584.0090) .. (668.7940,583.4677) --
      (668.7940,577.5877) -- cycle(673.2940,586.0490) .. controls
      (673.2940,588.2690) and (673.3540,590.7890) .. (673.5940,592.6490) --
      (669.4527,592.6490) -- (669.1552,589.5890) -- (668.9752,589.5890) .. controls
      (667.7152,591.8090) and (665.6740,593.1290) .. (663.1540,593.1290) .. controls
      (659.3140,593.1290) and (656.3140,589.8890) .. (656.3140,584.8490) .. controls
      (656.3140,577.5277) and (662.1340,574.3477) .. (668.6740,574.2865) --
      (668.6740,573.3877) .. controls (668.6740,569.4865) and (667.6540,566.8465) ..
      (664.0527,566.8465) .. controls (662.2540,566.8465) and (660.6327,567.3865) ..
      (659.2527,568.3465) -- (658.2328,565.1065) .. controls (659.4928,564.2065) and
      (662.1340,563.1865) .. (665.0153,563.1865) .. controls (670.8353,563.1865) and
      (673.2940,566.9665) .. (673.2940,573.4465) -- (673.2940,586.0490) -- cycle;
    \path[fill=ctablue,nonzero rule] (681.5747,563.6063) -- (685.4735,580.2876) ..
      controls (685.9547,582.5676) and (686.2547,584.4288) .. (686.6747,586.2876) --
      (686.7947,586.2876) .. controls (687.0947,584.6076) and (687.5735,582.4476) ..
      (687.9935,580.3476) -- (691.4747,563.6063) -- (696.0347,563.6063) --
      (691.2935,582.2676) .. controls (689.4947,589.5288) and (687.4535,596.5488) ..
      (684.0947,600.9901) .. controls (682.0535,603.5688) and (679.8922,604.8901) ..
      (678.8747,605.3701) -- (677.1335,601.4701) .. controls (678.5135,600.6901) and
      (679.9535,599.4301) .. (681.2747,597.8088) .. controls (682.4735,596.2488) and
      (683.3735,594.3288) .. (683.8522,592.8888) .. controls (684.0947,592.2288) and
      (684.2135,591.8688) .. (684.2135,591.4488) .. controls (684.2135,591.0888) and
      (684.1535,590.7288) .. (684.0335,590.3688) -- (676.7135,563.6063) --
      (681.5747,563.6063) -- cycle;
\end{scope}
\end{tikzpicture}
}

% render the logo for the footer only once (to speed up):
\newsavebox{\ctafootlogo}
\savebox{\ctafootlogo}{
	\resizebox{0.7cm}{!}{\ctalogo}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% chapter, section headings - nicer, maybe for the future
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%\RequirePackage[Bjornstrup]{fncychap}
%
%\renewcommand\DOCH{%
  %\settowidth{\py}{\CNoV\thechapter}
  %\addtolength{\py}{-10pt}
  %\fboxsep=0pt%
  %\colorbox{ctalightblue}{\rule{0pt}{40pt}\parbox[b]{\textwidth}{\hfill}}%
  %\kern-\py\raise20pt%
  %\hbox{\color{ctablue}\CNoV\thechapter}\\%
%}
%
%\renewcommand\DOTI[1]{%
  %\nointerlineskip\raggedright%
  %\fboxsep=\myhi%
  %\vskip-1ex%
  %\colorbox{ctalightblue}{\parbox[t]{\mylen}{\CTV\FmTi{#1}}}\par\nobreak%
  %\vskip 40pt%
%}
%
%\renewcommand\DOTIS[1]{%
  %\fboxsep=0pt
  %\colorbox{ctalightblue}{\rule{0pt}{40pt}\parbox[b]{\textwidth}{\hfill}}\\%
  %\nointerlineskip\raggedright%
  %\fboxsep=\myhi%
  %\colorbox{ctalightblue}{\parbox[t]{\mylen}{\CTV\FmTi{#1}}}\par\nobreak%
  %\vskip 40pt%
 %}
%
%%\usepackage[Bjornstrup]{fncychap}
%
%\ChTitleVar{\raggedleft\huge\sffamily\bfseries\color{ctablue}\thechapter\ }	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% chapter, section headings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\chapter{\FloatBarrier\if@openright\cleardoublepage\else\clearpage\fi\vspace*{0.85cm}\@startsection 
{chapter}
{0}
{\z@}%               
%{\vspace*{1cm}}
{-\baselineskip}
{7.6ex \@plus.2ex}%
{\FloatBarrier\normalfont\huge\bfseries\ctablue}}

\ifbool{docstyle}{}{
	\patchcmd{\chapter}{\if@openright\cleardoublepage\else\clearpage\fi}{}{}{} %% This allows chapters to begin without clearing pages
	\patchcmd{\chapter}{\thispagestyle{\chapterpagestyle}}{\thispagestyle{fancy}}{}{} %% This allows chapters to begin without starting a new page
}

\ifbool{docstyle}{}{
	\patchcmd{\references}{\if@openright\cleardoublepage\else\clearpage\fi}{}{}{} %% This allows references to begin without clearing pages
	\patchcmd{\references}{\thispagestyle{\chapterpagestyle}}{\thispagestyle{fancy}}{}{} %% This allows references to begin without starting a new page
}

\renewcommand{\section}{\@startsection
{section}%                   % the name
{1}%                         % the level
{0mm}%                       % the indent
{-\baselineskip}%            % the before skip
{0.5\baselineskip}%          % the after skip
{\normalfont\Large\bfseries\ctablue}} % the style

\renewcommand{\subsection}{\@startsection
{subsection}%                   % the name
{2}%                         % the level
{0mm}%                       % the indent
{-\baselineskip}%            % the before skip
{0.4\baselineskip}%          % the after skip
{\normalfont\Large\ctablue}} % the style

\renewcommand{\subsubsection}{\@startsection
{subsubsection}%                   % the name
{3}%                         % the level
{0mm}%                       % the indent
{-\baselineskip}%            % the before skip
{0.3\baselineskip}%          % the after skip
{\normalfont\large\ctablue}} % the style


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% page length
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\setlength{\parindent}{0in}
\setlength{\parskip}{1em}
\setlength{\textheight}{24.6cm}




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Header & Footer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifnum\value{foot}=0 \project{CTA Construction Project}
\fi
\ifnum\value{foot}=1 \project{CTAO gGmbH}
\fi
\ifnum\value{foot}=2 \project{CTA Consortium}
\fi
\ifnum\value{foot}=3 \project{GCT Sub-Consortium}
\fi


\fancypagestyle{firststyle}{
	\renewcommand{\headrulewidth}{0pt}
	\fancyhf{}
	\fancyfoot[L]{
								\setlength{\unitlength}{1mm} 
								\begin{picture}(0,0)
								\put(-1.3,-3){\usebox{\ctafootlogo}}
								\end{picture}
								\hspace{7mm}\scriptsize{\@project}
								\newline \put(7.975,-0.1){\scriptsize{\@shorttitle}}
								}
	\fancyfoot[R] {
									\scriptsize{\@acronym\ $|$\ v. \@version\ $|$\ \@realdate \hspace*{0.7mm}}
									\\ \scriptsize{\@addref}
								}		
	\fancyfoot[C] {\scriptsize{Page \thepage\ of \pageref{LastPage}}}
}


\fancypagestyle{plain}{
\fancyhf{}
\fancyhead[L]{{\scriptsize\nouppercase\leftmark}}
\fancyhead[R]{{\scriptsize\nouppercase\rightmark}}
\renewcommand{\footrulewidth}{0.2pt}
\fancyfoot[L]{\setlength{\unitlength}{1mm} 
		\begin{picture}(0,0)
				\put(-1.3,-3){\usebox{\ctafootlogo}}
		\end{picture}
		\hspace{7mm}\scriptsize{\@project}
	  \newline \put(7.975,-0.1){\scriptsize{\@shorttitle}}
		}
	\fancyfoot[R] {
									\scriptsize{\@acronym\ $|$\ v. \@version\ $|$\ \@realdate \hspace*{0.7mm}}
									\\ \scriptsize{\@addref}
								}							
\fancyfoot[C] {\scriptsize{Page \thepage\ of \pageref{LastPage}}}
}


\fancyhf{}
\pagestyle{fancy}

\fancyhead[L]{{\scriptsize\nouppercase\leftmark}}
\fancyhead[R]{{\scriptsize\nouppercase\rightmark}}

\renewcommand{\footrulewidth}{0.2pt}

\fancyfoot[L]{\setlength{\unitlength}{1mm} 
		\begin{picture}(0,0)
				\put(-1.3,-3){\usebox{\ctafootlogo}}
		\end{picture}
		\hspace{7mm}\scriptsize{\@project}
	  \newline \put(7.975,-0.1){\scriptsize{\@shorttitle}}
		}
	\fancyfoot[R] {
									\scriptsize{\@acronym\ $|$\ v. \@version\ $|$\ \@realdate \hspace*{0.7mm}}
									\\ \scriptsize{\@addref}
								}								
\fancyfoot[C] {\scriptsize{Page \thepage\ of \pageref{LastPage}}}		

	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Titlepage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifbool{docstyle}{
	\renewcommand{\maketitle}{%
	\newpage

	\thispagestyle{firststyle}

	\vspace*{0.1cm}	


	\begin{center}
					\resizebox{7cm}{!}{\ctalogo}
	\end{center}

	\vspace{2cm}	


	\begin{center}
	\hrule
	\vspace{2mm}
		\ctablue{\Huge{\textbf{\@title}}}
	\vspace{4mm}
	\hrule
	\end{center}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% a picture or logos
	%%\vfill
	%%\begin{center}
		%%\includegraphics[trim=0cm 0cm 0cm 0cm, clip= true, width=\textwidth]%%{GCT.png}
	%%\end{center}
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	\color{ctadarkgray}	



	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Document Properties Page
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	\newpage

	\pagestyle{fancy}
	\markboth{Document Properties}{} 


	\begingroup
	\renewcommand{\arraystretch}{1.5}
	%\fontsize{9pt}{1em}
	%\small
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% This Version
	\begin{center}

		\begin{tabular}
		{|p{0.0375\textwidth}|p{0.1125\textwidth}|p{0.295\textwidth}|p{0.1031\textwidth}|>{\footnotesize\openup=2.5pt}p{0.0534\textwidth} p{0.235\textwidth}|}
		\hline 
		\multicolumn{6}{|l|}{\cellcolor{ctalightblue} \textbf{This Version:} }\\
		\hline
		\cellcolor{ctalightblue}Ver. &\cellcolor{ctalightblue} Created &\cellcolor{ctalightblue} Comment & \cellcolor{ctalightblue}Distribution & \multicolumn{2}{|l|}{\cellcolor{ctalightblue}Corresponding...} \\
		\hline
		\@version & \tabledate\@realdate & \@comment & \@distribution & Editor: \newline Checker: \newline approver: & \@editor \newline  \underline{\hspace{3.7cm}} \newline \underline{\hspace{3.7cm}} \\
		\hline
		\end{tabular}
	\end{center}

	\vspace{0.5cm}


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Keywords	
	\begin{center}
		\begin{tabular}{|p{0.0375\textwidth} !{\color{white}\vrule} p{0.1125\textwidth}!{\color{white}\vrule} p{0.1\textwidth}!{\color{white}\vrule} p{0.1031\textwidth}!{\color{white}\vrule} p{0.0534\textwidth} p{0.1\textwidth}|}
		\hline 
		\multicolumn{6}{|l|}{\cellcolor{ctalightblue} \textbf{Keywords:} }\\
		\hline
		\multicolumn{6}{|p{0.972\textwidth}|}{\@keywords \hfill } \\
		 & & & & & \\[1pt]
		\hline
		\end{tabular}
	\end{center}
	\vspace{0.5cm}


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Previous Versions

	\setlength\LTleft{0pt}
	\setlength\LTright{0pt}

		\begin{longtable}{|p{0.0375\textwidth}|p{0.1125\textwidth}|p{0.295\textwidth}|p{0.1031\textwidth}|>{\footnotesize\openup=2.5pt}p{0.0534\textwidth} p{0.235\textwidth}|}
		\hline 
		\multicolumn{6}{|l|}{\cellcolor{ctalightblue} \textbf{Version History:} }\\
		\hline
		\cellcolor{ctalightblue}Ver. &\cellcolor{ctalightblue} Date&\cellcolor{ctalightblue} Comment & \cellcolor{ctalightblue}Distribution & \multicolumn{2}{|l|}{\cellcolor{ctalightblue} Corresponding...} \\
		\hline

		\@history
		
		\end{longtable}
	%}	
	%\end{center}

	\endgroup
		
	}

	\color{ctadarkgray}
}{


%%%%%%%%%%%%%%%%%%%%%%%%%%% now follows the title page for short version %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\maketitle}{%
	\newpage

	\thispagestyle{firststyle}

	%\vspace*{0.1cm}	

	\begin{center}
					\resizebox{7cm}{!}{\ctalogo}
	\end{center}

	\vspace{1cm}	

	\begin{center}
	\hrule
	\vspace{2mm}
		\ctablue{\Huge{\textbf{\@title}}}
	\vspace{4mm}
	\hrule
	\end{center}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% a picture or logos
	%\vfill
	%\begin{center}
		%\includegraphics[trim=0cm 0cm 0cm 0cm, clip= true, width=\textwidth]{GCT.png}
	%\end{center}
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	\color{ctadarkgray}	

	\vspace{1cm}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%% Document Properties Page - now also on the first page
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	\pagestyle{fancy}
	\markboth{Document Properties}{} 


	\begingroup
	\renewcommand{\arraystretch}{1.5}
	%\fontsize{9pt}{1em}
	%\small
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% This Version
	\begin{center}

		\begin{tabular}
		{|p{0.0375\textwidth}|p{0.1125\textwidth}|p{0.26\textwidth}|p{0.11\textwidth}|p{0.16\textwidth}|p{0.16\textwidth}|}
		\hline 
		\multicolumn{6}{|l|}{\cellcolor{ctalightblue} \textbf{Current Release:} }\\
		\hline
		\cellcolor{ctalightblue}Ver. &\cellcolor{ctalightblue}Created &\cellcolor{ctalightblue}Comment &\cellcolor{ctalightblue}Distribution & \cellcolor{ctalightblue}Editor & \cellcolor{ctalightblue}Approver(s) \\
		\hline
	         \@version & \tabledate\@realdate & \@comment & \@distribution & \@editor & \@approver \\
		\hline
		\end{tabular}
	\end{center}

	%\vspace{0.1cm}


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Keywords	
	%%\begin{center}
	%%	\begin{tabular}{|p{0.0375\textwidth} !{\color{white}\vrule} p{0.1125\textwidth}!{\color{white}\vrule} p{0.1\textwidth}!{\color{white}\vrule} p{0.1031\textwidth}!{\color{white}\vrule} p{0.0534\textwidth} p{0.1\textwidth}|}
	%%	\hline 
	%%	\multicolumn{6}{|l|}{\cellcolor{ctalightblue} \textbf{Keywords:} }\\
	%%	\hline
	%%	\multicolumn{6}{|p{0.972\textwidth}|}{\@keywords \hfill } \\
	%%	 & & & & & \\[1pt]
	%%	\hline
	%%	\end{tabular}
	%%\end{center}
	%\vspace{0.2cm}


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Previous Versions

	\setlength\LTleft{0pt}
	\setlength\LTright{0pt}

		\begin{longtable}		
		{|p{0.0375\textwidth}|p{0.1125\textwidth}|p{0.26\textwidth}|p{0.11\textwidth}|p{0.16\textwidth}|p{0.16\textwidth}|}
		\hline 
		\multicolumn{6}{|l|}{\cellcolor{ctalightblue} \textbf{History:} }\\
		\hline
		\cellcolor{ctalightblue}Ver. &\cellcolor{ctalightblue}Created &\cellcolor{ctalightblue}Comment &\cellcolor{ctalightblue}Distribution & \cellcolor{ctalightblue}Editor & \cellcolor{ctalightblue}Approver(s) \\		
		\hline 
		\@history
		\end{longtable}
	%}	
	%\end{center}

	\endgroup
		
	}

	\color{ctadarkgray}

}
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% ToC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\maketoc}{
\FloatBarrier\if@openright\cleardoublepage\else\clearpage\fi
\color{ctablue}
  \renewcommand{\contentsname}{\ctablue{Table of Contents}}
	\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
	\setlength\cftparskip{3pt}
	\setcounter{tocdepth}{1}
  \addcontentsline{toc}{chapter}{Table of Contents}
	\markboth{Table of Contents}{}
	\tableofcontents

\color{ctadarkgray}	
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Redefine left and rightmark
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}	


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% table style
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewenvironment{table}
    {\renewcommand{\arraystretch}{1.4}%
	\arrayrulecolor{white}
	    \@float{table}}
    {\end@float}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Pdf metadata
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\AtBeginDocument{
\hypersetup{
    pdftitle={\@shorttitle},    % title
    pdfauthor={\@editor},     % author
    pdfsubject={CTA document template},   % subject of the document
    pdfcreator={CTA template by Besel, Goullon},   % creator of the document
    pdfproducer={LaTeX}, % producer of the document
    pdfkeywords={\@keywords}, % list of keywords
		urlcolor     = ctablue, %Colour for external hyperlinks
		linkcolor    = ctablue, %Colour of internal links
		citecolor    = ctablue  %Colour of citations
}
}