#!/usr/bin/env python

import sys
from datetime import datetime as dt

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split(',')
    # increase counters
    try:
        starttime=dt.strptime(words[1],'%Y-%m-%d %H:%M:%S')
        year=starttime.year
        month=starttime.month
        day=starttime.day
        dow=starttime.weekday()
        print '%s,%s,%s,%s,%s'%(line,year,month,day,dow)
    except:
        pass
