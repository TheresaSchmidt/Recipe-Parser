POS tags (STTS) are included in the corpus but were not used for parsing. They were annotated with the ParZu parser (https://github.com/rsennrich/ParZu).

Excerpt from the corpus (in CoNLL-U format):

ID	FORM 	_	POS	LABEL	_	HEAD	DEPREL	DEPRELS	_

82	Dann    _       ADVO    O       _       0       root    _	_ 
83	etwa	_	ADVO	O	_	0	root	_	_ 
84	drei	_	CARD	B-Bedingung	_	86	Zeitangabe	_	_ 
85	Minuten	_	NN	L-Bedingung	_	86	Zeitangabe	_	_ 
86	cremig	_	ADJD	B-Kochschritt	_	91	Nullanapher	_	_ 
87	rühren	_	VVINF	L-Kochschritt	_	91	Nullanapher	_	_ 
88	.	_	$.O	O	_	0	root	_	_ 
89	Den	_	ART	B-Zutat	_	91	Input	_	_ 
90	Rum	_	NN	L-Zutat	_	91	Input	_	_ 
91	unterrühren	_	VVINF	U-Kochschritt	_	100	Nullanapher	_	_ 



Excerpt in ConLL2003 format (= only labels, no relations):

TOKEN	POS	O	LABEL

Dann	ADV	O	O
etwa	ADV	O	O
drei	CARD	O	B-Bedingung
Minuten	NN	O	L-Bedingung
cremig	ADJD	O	B-Kochschritt
rühren	VVINF	O	L-Kochschritt
.	$.	O	O

Den	ART	O	B-Zutat
Rum	NN	O	L-Zutat
unterrühren	VVINF	O	U-Kochschritt
.	$.	O	O
