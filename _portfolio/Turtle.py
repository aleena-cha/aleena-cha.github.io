---
title: "Turtle"
excerpt: "This script uses Python's turtle and colorsys libraries to draw a geometric floral mandala with 30 concentric layers of 18 petals each on a pastel blue background (#d0d9ff). It gradually shrinks the petal size in each layer while shifting through a rainbow HSV-to-RGB color gradient from dark to bright.<br/><img src='/images/Turtle.png'>"
codeurl: 'https://github.com/aleena-cha/aleena-cha.github.io/blob/master/_layouts/Turtle.py'
collection: portfolio
---

Python 3.13.15 (tags/v3.13.15:4061bc4, Aug  5 2026, 13:05:39) [MSC v.1944 64 bit (AMD64)] on win32<br>
Enter "help" below or click "Help" above for more information.<br>
>>> import turtle as tur<br>
>>> import colorsys as cs<br>
>>> tur.setup(900, 900)<br>
>>> tur.speed(0)<br>
>>> tur.width(1)<br>
>>> tur.bgcolor("#d0d9ff")<br>
>>> for j in range(30):<br>
...     for i in range(18):<br>
...         tur.color(cs.hsv_to_rgb(i / 18, 0.8, j / 30))<br>
...         tur.right(90)<br>
...         tur.circle(210 - j * 5, 90)<br>
...         tur.left(90)<br>
...         tur.circle(210 - j * 5, 90)<br>
...         tur.right(180)<br>
...         tur.circle(50, 20)<br>
... 
...         
tur.hideturtle()
>>> tur.done()
