#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:07:11 2019

@author: menghao
"""

import os
from removebg import RemoveBg
from PIL import Image, ImageDraw, ImageFont
'''parameters'''
rmbg = RemoveBg('xhRvd2MUMUX4H5ESV8hwsvJP', 'error.log')
path = '/Users/menghao/Document/z_others/pics/'


'''image-matting'''
for pic in os.listdir(path):
    rmbg.remove_background_from_img_file(path + str(pic))
    
    if str(pic) != '.DS_Store':
        '''change bg color'''
        img = Image.open(path + str(pic) + '_no_bg.png')
        x, y = img.size
        try:
            blue = Image.new('RGBA', img.size, (0,0,255))
            blue.paste(img, (0,0,x,y), img)
            blue.save(path + str(pic) + '_' + 'blue' + '.png')
            
            red = Image.new('RGBA', img.size, (255,0,0))
            red.paste(img, (0,0,x,y), img)
            red.save(path + str(pic) + '_' + 'red' + '.png')
                        
            '''add water mark'''
            img_r = Image.open(path + str(pic) + '_' + 'red' + '.png')
            img_b = Image.open(path + str(pic) + '_' + 'blue' + '.png')
            txt = Image.new('RGBA', img.size, (0,100,0,0))
            fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 30)
            mark = ImageDraw.Draw(txt)
            mark.text((txt.size[0]-160, txt.size[1]-50), 'Wu\'swork', font=fnt, fill=(255,255,255,255))
            out = Image.alpha_composite(img_r, txt)
            out.save(path + 'watermark/' + str(pic) + '_r' + 'mark' + '.png')
            out = Image.alpha_composite(img_b, txt)
            out.save(path + 'watermark/' + str(pic) + '_b' + 'mark' + '.png')
        except:
            pass
        
