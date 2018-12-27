#!/bin/bash
  dt=`date '+%Y%m%d%H%M'`
  l=10
  # CASA1.PNG
  python generate.py -hf data/casa1.png -bf data/refugee_camp_before_data.jpg -date $dt -l $l -r 80 -s 1.2 -d 20x5 -margin 0
  python generate.py -hf data/casa1.png -bf data/Desert1.png -date $dt -l $l -s 3 -r 70 -d 0x0 -h 35 -margin 0 -m
  # python generate.py -hf data/casa1.png -bf data/Desert2.png no
  # python generate.py -hf data/casa1.png -bf data/Desert3.png no
  python generate.py -hf data/casa1.png -bf data/Desert4.png -date $dt -l $l -s 2.2 -r 150 -d 0x0 -h 50 -margin 0
  python generate.py -hf data/casa1.png -bf data/desert5.png -date $dt -l $l -s 4 -r 160 -d 0x0 -margin 0
  # python generate.py -hf data/casa1.png -bf data/Desert6.png no
  python generate.py -hf data/casa1.png -bf data/Desert7.png -date $dt -l $l -s 1.5 -r 75 -d 0x0 -h 60 -margin 0

  # CASA2.PNG no mola
  # CASA3.JPG
  python generate.py -hf data/casa3.jpg -bf data/refugee_camp_before_data.jpg -date $dt -l $l -r 80 -s 1.2 -d 20x5 -margin 0
  # python generate.py -hf data/casa3.jpg -bf data/Desert1.png -date $dt -l $l
  # python generate.py -hf data/casa3.jpg -bf data/Desert2.png -date $dt -l $l
  # python generate.py -hf data/casa3.jpg -bf data/Desert3.png -date $dt -l $l
  # python generate.py -hf data/casa3.jpg -bf data/Desert4.png -date $dt -l $l
  # python generate.py -hf data/casa3.jpg -bf data/Desert5.png -date $dt -l $l
  # python generate.py -hf data/casa3.jpg -bf data/Desert6.png -date $dt -l $l
  # python generate.py -hf data/casa3.jpg -bf data/Desert7.png -date $dt -l $l

  # CASA4.JPG
  python generate.py -hf data/casa4.jpg -bf data/refugee_camp_before_data.jpg -date $dt -l $l -r 80 -s 1.2 -d 20x5 -margin 0
  # python generate.py -hf data/casa4.jpg -bf data/Desert1.png -date $dt -l $l
  # python generate.py -hf data/casa4.jpg -bf data/Desert2.png -date $dt -l $l
  # python generate.py -hf data/casa4.jpg -bf data/Desert3.png -date $dt -l $l
  # python generate.py -hf data/casa4.jpg -bf data/Desert4.png -date $dt -l $l
  # python generate.py -hf data/casa4.jpg -bf data/Desert5.png -date $dt -l $l
  # python generate.py -hf data/casa4.jpg -bf data/Desert6.png -date $dt -l $l
  # python generate.py -hf data/casa4.jpg -bf data/Desert7.png -date $dt -l $l

  # CASA5.JPG millor no
  # ORIGINAL.PNG
  python generate.py -hf data/original.png -bf data/refugee_camp_before_data.jpg -date $dt -l $l -r 80 -s 1.2 -d 20x5 -margin 0
  # python generate.py -hf data/original.png -bf data/Desert1.png -date $dt -l $l
  # python generate.py -hf data/original.png -bf data/Desert2.png -date $dt -l $l
  # python generate.py -hf data/original.png -bf data/Desert3.png -date $dt -l $l
  # python generate.py -hf data/original.png -bf data/Desert4.png -date $dt -l $l
  # python generate.py -hf data/original.png -bf data/Desert5.png -date $dt -l $l
  # python generate.py -hf data/original.png -bf data/Desert6.png -date $dt -l $l
  # python generate.py -hf data/original.png -bf data/Desert7.png -date $dt -l $l

  exit 1
