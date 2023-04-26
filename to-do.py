# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:47:50 2023

@author: sophie_bauchinger
"""

"""    TO-DO
punkte auswählen lassen wenn sie strat / trop sind oder troposphärische outlier

trop outlier identifiziert - ! muster-erkennung ! 
= gleiches problem wie katharina, sie hat halt andere daten und muster, aber gleiches problem
bei mir: asiatischer monsoon einfluss is literatur wissen, aber wo sind die proben? 
haben auch andere muster: outlier in A is nie / immer ... outlier in B -> nur in geographischem bereich ... ? 

wir fangen mit der verteilung von der nicht-outliers an
böhnisch paper als anfang - in die richtung? 

plots zu pot. K mit abstand zu tropopause auf Y, eq. lat in X
-> treffen mit moritz: modell daten, im meeting fragen ob das so einfach geht, wies sinnvoll wird? 

# unterschiede in der verteilung von GHG in UTLS über seasons + HCF files (auch langlebige gase) 
langlebig: cfcs, hcfcs, ... 
(nicht hfo files, sind kurzlebig und haben low gwp)
mit den klassischen 4 anfangen

daten jahreszeitlich gruppieren
hat sich in den 15 jahren was verändert? (erstmals alle jahre gleich behandeln)


(böhnisch): altersbestimmung sf6 oder co2 unterschiede ? 
n20 kann man nicht so vergleichen
sf6/co2 haben gute gradiented: machen die in der seasonality das gleiche? wie sehen die trends aus? geographic variability? 
in den N extrop: längenabhängigkeit mit jahreszeiten? zeitliche veränderung über die jahre? 

Sven page 35 nachbauen
SPURT paper: diagonale isolinien

y axis: theta oder delta theta: beide optionen einbauen
delta theta hat verschiedene tropopausen definitionen (weitermachen wo sven aufgehört hat)
pv isolines von wo? 

de-trending wegen cont. increase (bzw more complex for methane) - wrt. mauna loa mean (free troposphere mean value)

emac modell (moritz): aus dem modell pv isolinien  bekommen, später auch clams daten 


! 1. Sven 2D plot
    draus lernen
c_plot: pl_gradient_by_season (sven fig 32 oder so), plot_box_2d

! Vertikalprofile aufgeschlüsselt
2. gradienten (line plot)
    a. zeitlich aufschlüsseln
    b. CO2, CH4 spezifically
    c. seasonale zyklus nach oben propagiert (CO2, CH4) - see Diallo paper
    transport-bedingt für n2o, sf6: fixer trop. abstand, sehen wie ne seasonality? tbd!
3. Wie die 4 gase jahreszeitlich mit correlation zu ozon variieren (also ozon als höhenkoordinate, 
                                                                    chemische tropopause - braucht 
                                                                    langzeitmessungen zum kalibrieren. variiert seasonally, gibts nen trend in der correlation) 

ozon zeug macht nur in stratosphere sinn, weil sonst sehr variabel
O3 - CO coordinate system: spurengase als kurve darin, kurve ändert sich jahreszeitlich 
(nicht sinnvoll: sf6 und N2O nicht hyperbolisch) - nur wenn strat / trop übergang
ABER correlation mit ozon hat nen jahreszeitlichen gradienten, den wollen wir uns anschaun        

""" 