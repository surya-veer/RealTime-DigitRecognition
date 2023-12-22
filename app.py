import pygame
from process_image import get_output_image

# Predefined colors, pen radius, and font color
black = [0, 0, 0]
white = [255, 255, 255]
red = [255, 0, 0]
green = [0, 255, 0]
light_gray = (200, 200, 200)
draw_on = False
last_pos = (0, 0)
color = (255, 128, 0)
radius = 7
font_size = 500

# Image size
width = 640
height = 640

# Initializing screen
screen = pygame.display.set_mode((width * 2, height))
screen.fill(white)
pygame.font.init()


def show_output_image(img):
    surf = pygame.pixelcopy.make_surface(img)
    surf = pygame.transform.rotate(surf, -270)
    surf = pygame.transform.flip(surf, 0, 1)
    screen.blit(surf, (width + 2, 0))


def crope(orginal):
    cropped = pygame.Surface((width - 5, height - 5))
    cropped.blit(orginal, (0, 0), (0, 0, width - 5, height - 5))
    return cropped


def roundline(srf, color, start, end, radius=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color, (x, y), radius)


def draw_partition_line():
    pygame.draw.line(screen, black, [width, 0], [width, height], 8)


def display_options_screen():
    font = pygame.font.Font(None, 36)
    options = ["Simple NN + Sigmoid", "Simple NN + Softmax", "CNN","KNN"]
    button_height = 50
    buttons = []

    for i, option in enumerate(options):
        button = pygame.Rect(width // 2 - 100, 100 + i * button_height, 200, 40)
        buttons.append(button)
        
        # Check if the mouse is inside the button's rectangle
        if button.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(screen, light_gray, button)  # Change color when hovering
        else:
            pygame.draw.rect(screen, white, button)
        
        text = font.render(option, True, black)
        screen.blit(text, (button.x + button.width // 2 - text.get_width() // 2,
                        button.y + button.height // 2 - text.get_height() // 2))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise StopIteration
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, button in enumerate(buttons):
                    if button.collidepoint(event.pos):
                        return i + 1  # Return the selected option

# Main loop
selected_option = None

try:
    # Display options screen only once
    font = pygame.font.Font(None, 36)
    options = ["Simple NN + Sigmoid", "Simple NN + Softmax", "CNN","KNN"]
    button_height = 50
    buttons = []

    for i, option in enumerate(options):
        button = pygame.Rect(width // 2 - 100, 100 + i * button_height, 200, 40)
        buttons.append(button)
        
        # Kiểm tra nếu chuột nằm trong hình chữ nhật của nút
        if button.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(screen, light_gray, button)  # Màu sắc thay đổi khi hover
        else:
            pygame.draw.rect(screen, white, button)
        
        text = font.render(option, True, black)
        screen.blit(text, (button.x + button.width // 2 - text.get_width() // 2,
                        button.y + button.height // 2 - text.get_height() // 2))


    pygame.display.flip()

    while selected_option is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise StopIteration
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, button in enumerate(buttons):
                    if button.collidepoint(event.pos):
                        selected_option = i + 1  # Set the selected option
                        screen.fill(white)  # Clear the screen
                        
    # Continue with the selected option
    while True:
        e = pygame.event.wait()
        draw_partition_line()

        # clear screen after right click
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
            screen.fill(white)

        # quit
        if e.type == pygame.QUIT:
            raise StopIteration

        # start drawing after left click
        if e.type == pygame.MOUSEBUTTONDOWN and e.button != 3:
            color = black
            pygame.draw.circle(screen, color, e.pos, radius)
            draw_on = True

        # stop drawing after releasing left click
        if e.type == pygame.MOUSEBUTTONUP and e.button != 3:
            draw_on = False
            fname = "out.png"

            img = crope(screen)
            pygame.image.save(img, fname)

            output_img = get_output_image(fname, selected_option)
            show_output_image(output_img)

        # start drawing line on screen if draw is true
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radius)
                roundline(screen, color, e.pos, last_pos, radius)
            last_pos = e.pos

        pygame.display.flip()

except StopIteration:
    pass

pygame.quit()
