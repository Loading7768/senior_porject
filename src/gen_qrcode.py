import qrcode

img = qrcode.make("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
img.save("../outputs/qrcode.png")
