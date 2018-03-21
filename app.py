###############################################################
#########          By: suryaveer @IIT Indore         ##########
#########     GITHUB: https://github.com/surya-veer  ##########
###############################################################

import pygame


#select model for prediction
# MODEL = 'SVC'
MODEL = 'KERAS'

# trained on 8X8 images 
if(MODEL=='SVC'):
	from models_svc import digit_recognition

# trained on 28X28 images 
if(MODEL=='KERAS'):
	from models_keras import digit_recognition





#pre defined colors, pen radius and font color
black = [0,0,0]
white = [255, 255, 255]
red = [255, 0, 0]
green = [0,255,0]
draw_on = False
last_pos = (0, 0)
color = (255, 128, 0)
radius = 20
font_size = 500

#initializing screen 
screen = pygame.display.set_mode((1000,500))
screen.fill(white)
pygame.font.init() 

def print_digit(num):
    #line for partition
    pygame.draw.rect(screen, white, [500, 0, 1000, 500])


    #heading text
    heading_font = pygame.font.SysFont('', 100)
    heading = heading_font.render('Output is', False, green)
    screen.blit(heading,(600,20))

    #output text
    myfont = pygame.font.SysFont('',font_size)
    output_text = myfont.render(str(num)[1], False, red)
    screen.blit(output_text,(680,130))


def crope(orginal):
    cropped = pygame.Surface((500, 500))
    cropped.blit(orginal, (0, 0), (0,0, 500, 500))
    return cropped

def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)

def draw_line():
    pygame.draw.line(screen, black, [510, 0], [510, 500], 10)

try:
    while True:
        # get all events
        e = pygame.event.wait()
        draw_line()

        #clear screen after right click
        if(e.type == pygame.MOUSEBUTTONDOWN and e.button==3):
            screen.fill(white)

        #quit
        if e.type == pygame.QUIT:
            raise StopIteration

        #start drawing after left click
        if(e.type == pygame.MOUSEBUTTONDOWN and e.button!=3):
            color = black
            pygame.draw.circle(screen, color, e.pos, radius)
            draw_on = True

        #stop drawing after releasing left click
        if e.type == pygame.MOUSEBUTTONUP and e.button!=3:
            draw_on = False
            fname = "assets/out.png"

            img = crope(screen)
            pygame.image.save(img, fname)
            res = digit_recognition.check()
            print("Output is:=>>  ",res[0], '\n\n Getting...')
            print_digit(res)

        #start drawing line on screen if draw is true
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radius)
                roundline(screen, color, e.pos, last_pos,  radius)
                
            last_pos = e.pos

        pygame.display.flip()

except StopIteration:
    pass

pygame.quit()
