# import the Pillow library to use its Image, ImageDraw, and ImageFont classes
from PIL import Image, ImageDraw, ImageFont
import random
import os


def generate_training_data():

    # remove images directory if it exists, and remove the training_data.csv file
    if os.path.exists('images'):
        os.system('rm -rf images')
    if os.path.exists('training_data.csv'):
        os.system('rm training_data.csv')

    # creart directory to store the images, if it doesn't exist already
    if not os.path.exists('images'):
        os.makedirs('images')

    image_width = 5
    image_height = 7
    font_size = 7
    font_size_ar = 8
    starting_position = (0, 0)

    number_map = {
        0: '٠',
        1: '١',
        2: '٢',
        3: '٣',
        4: '٤',
        5: '٥',
        6: '٦',
        7: '٧',
        8: '٨',
        9: '٩'
    }

    # create an empty array to store the images
    images = []

    # create a font object using the the simplest font
    # font = ImageFont.truetype('./fonts/Arial.ttf', font_size)
    font = ImageFont.truetype(
        '/System/Library/Fonts/Supplemental/Courier New.ttf', font_size)

    font_ar = ImageFont.truetype(
        '/System/Library/Fonts/Supplemental/Arial.ttf', font_size_ar)

    # open a file to write the pixel values to, a csv file
    with open('training_data.csv', 'w') as f:

        # loop through the numbers 0 to 9
        for num in range(10):
            # create a new image with the desired size
            img = Image.new('1', (image_width, image_height), color=0)

            # create a draw object to draw on the image
            draw = ImageDraw.Draw(img)

            # draw the number on the image using the font
            draw.text(starting_position, str(num), font=font, fill=10)

            # save the image to a file
            img.save('images/number_{}.png'.format(num))

            # get the pixel values from the image
            pixel_values = img.getdata()

            # convert the pixel values to a 1-dimensional array
            pixel_array = list(pixel_values)

            # convert the pixels to a comma-separated string
            pixel_string = ','.join(str(pixel) for pixel in pixel_array)

            # add the num as a label to the pixel string at the end
            pixel_string += ',' + str(num)

            # write the pixel string to the file, followed by a new line
            f.write(pixel_string + '\n')  # without noise

            # generate 20 more lines with noise (3 random pixels set to 5)
            for i in range(20):
                # create a copy of the pixel list
                pixel_array_copy = pixel_array.copy()

                # set 3 random pixels to 5
                for j in range(3):
                    pixel_array_copy[random.randint(
                        0, len(pixel_array_copy) - 1)] = 5

                # convert the pixels to a comma-separated string
                pixel_string = ','.join(str(pixel)
                                        for pixel in pixel_array_copy)

                # add the num as a label to the pixel string at the end
                pixel_string += ',' + str(num)

                # write the pixel string to the file, followed by a new line
                f.write(pixel_string + '\n')

    # add to the file
    with open('training_data.csv', 'a') as f:

        # loop through the numbers 0 to 9
        for num in range(10):
            # create a new image with the desired size
            img = Image.new('1', (image_width, image_height), color=0)

            # create a draw object to draw on the image
            draw = ImageDraw.Draw(img)

            # draw the number on the image using the font
            draw.text(starting_position,
                      number_map[num], font=font_ar, fill=10)

            # save the image to a file
            img.save('images/number_{}_ar.png'.format(num))

            # get the pixel values from the image
            pixel_values = img.getdata()

            # convert the pixel values to a 1-dimensional array
            pixel_array = list(pixel_values)

            # convert the pixels to a comma-separated string
            pixel_string = ','.join(str(pixel) for pixel in pixel_array)

            # add the num as a label to the pixel string at the end
            pixel_string += ',' + str(num)

            # write the pixel string to the file, followed by a new line
            f.write(pixel_string + '\n')  # without noise

            # generate 20 more lines with noise (3 random pixels set to 5)
            for i in range(20):
                # create a copy of the pixel list
                pixel_array_copy = pixel_array.copy()

                # set 3 random pixels to 5
                for j in range(3):
                    pixel_array_copy[random.randint(
                        0, len(pixel_array_copy) - 1)] = 5

                # convert the pixels to a comma-separated string
                pixel_string = ','.join(str(pixel)
                                        for pixel in pixel_array_copy)

                # add the num as a label to the pixel string at the end
                pixel_string += ',' + str(num)

                # write the pixel string to the file, followed by a new line
                f.write(pixel_string + '\n')
