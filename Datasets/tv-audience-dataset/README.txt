The file tv-audience-dataset.csv contains the data relative to tv watching behaviour.

Columns are:
channel ID: channel id from 1 to 217.
slot: hour inside the week relative to the start of the view, from 1 to 24*7 = 168.
week: week from 1 to 19. Weeks 14 and 19 should not be used because they contain errors.
genre ID: it is the id of the genre, form 1 to 8. Genre/subgenre mapping is attached below.
subGenre ID: it is the id of the subgenre, from 1 to 114. Genre/subgenre mapping is attached below.
user ID: it is the id of the user.
program ID: it is the id of the program. The same program can occur multiple times (e.g. a tv show).
event ID: it is the id of the particular instance of a program. It is unique, but it can span multiple slots.
duration: duration of the view.

Genre/subgenre conversion table
genre	genreID	subgenre	subGenreID
movie	3	science fiction	51
movie	3	musical	63
movie	3	animation	17
movie	3	dramatic	29
movie	3	fantastic	46
movie	3	war	67
movie	3	funny	56
movie	3	adventure	4
movie	3	action	34
movie	3	comedy	3
movie	3	thriller	28
movie	3	cinema	7
movie	3	romantic	45
movie	3	biography	54
movie	3	western	19
movie	3	erotic	82
movie	3	short film	66
movie	3	horror	24
movie	3	crime	42
undefined	7	undefined 	47
other programs	8	other	112
other programs	8	other programs	50
other programs	8	educational	108
other programs	8	special events	107
other programs	8	shopping	96
entertainment	5	mini series	104
entertainment	5	TV soap	102
entertainment	5	series	92
entertainment	5	science fiction	65
entertainment	5	reality show	61
entertainment	5	soap opera	64
entertainment	5	animation	71
entertainment	5	show	13
entertainment	5	dramatic	74
entertainment	5	docu-fiction	109
entertainment	5	theatre	8
entertainment	5	exhibition	41
entertainment	5	talk show	80
entertainment	5	entertainment	81
entertainment	5	quiz	95
entertainment	5	sit com	36
entertainment	5	fiction	31
entertainment	5	game	98
society	4	cinema magazine	32
society	4	travel magazine	87
society	4	reportage	59
society	4	nature	30
society	4	magazine	60
society	4	topical	10
society	4	music	55
society	4	adventure	75
society	4	fishing	93
society	4	lifestyle	38
society	4	society	27
society	4	technology	84
society	4	religion	52
society	4	history	26
society	4	documentary	5
society	4	economics	111
society	4	hobby	91
society	4	sport	99
society	4	nature magazine	48
society	4	fashion	89
society	4	travels	23
society	4	culture magazine	25
society	4	communities	76
society	4	cinema	53
society	4	culture and society	69
society	4	science magazine	94
society	4	science	15
society	4	politics	62
society	4	cooking	37
society	4	art and culture	6
sport	2	winter sports	79
sport	2	other	20
sport	2	poker	58
sport	2	hockey	22
sport	2	cycling	21
sport	2	wrestling	83
sport	2	sport	43
sport	2	soccer	16
sport	2	equestrian	72
sport	2	rugby	44
sport	2	swimming	78
sport	2	athletics	68
sport	2	baseball	40
sport	2	football usa	39
sport	2	boxe	57
sport	2	basket	2
sport	2	motors	33
sport	2	golf	70
sport	2	sail	97
sport	2	skiing	14
sport	2	tennis	12
sport	2	volley	18
sport	2	handball	103
kids and music	1	series	85
kids and music	1	concert	106
kids and music	1	documentary	110
kids and music	1	animation movie	105
kids and music	1	videoclip	101
kids and music	1	dance	73
kids and music	1	magazine	77
kids and music	1	music	1
kids and music	1	games	100
kids and music	1	cartoons	35
kids and music	1	educational	49
kids and music	1	children	88
kids and music	1	null	114
information	6	news	9
information	6	economics	90
information	6	sport	86
information	6	weather forecast	113
information	6	newscast	11
