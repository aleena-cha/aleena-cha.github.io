---
title: "Turtle"
excerpt: "This script uses Python's turtle and colorsys libraries to draw a geometric floral mandala with 30 concentric layers of 18 petals each on a pastel blue background (#d0d9ff). It gradually shrinks the petal size in each layer while shifting through a rainbow HSV-to-RGB color gradient from dark to bright.<br/><img src='/images/Turtle.png'>"
codeurl: 'https://github.com/aleena-cha/aleena-cha.github.io/blob/master/files/Turtle.py'
collection: portfolio
---

Python 3.13.15 (tags/v3.13.15:4061bc4, Aug  5 2026, 13:05:39) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import turtle as tur
>>> import colorsys as cs
>>> tur.setup(900, 900)
>>> tur.speed(0)
>>> tur.width(1)
>>> tur.bgcolor("#d0d9ff")
>>> for j in range(30):
...     for i in range(18):
...         tur.color(cs.hsv_to_rgb(i / 18, 0.8, j / 30))
...         tur.right(90)
...         tur.circle(210 - j * 5, 90)
...         tur.left(90)
...         tur.circle(210 - j * 5, 90)
...         tur.right(180)
...         tur.circle(50, 20)
... 
...         
tur.hideturtle()
>>> tur.done()
