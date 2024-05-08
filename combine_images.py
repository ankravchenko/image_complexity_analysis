from PIL import Image


def combine_images(columns, space, images, name):
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([Image.open(image).width for image in images])
    height_max = max([Image.open(image).height for image in images])
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, image in enumerate(images):
        img = Image.open(image)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
    background.save(name)


#combine_images(columns=2, space=20, images=['/vol/tcm11/kravchenko/correct_rdms_1000_weighted/kagome_corr.png', '/vol/tcm11/kravchenko/correct_rdms_1000_weighted/triangle_corr.png', '/vol/tcm11/kravchenko/correct_rdms_1000_weighted/kagome_corr_weighted.png', '/vol/tcm11/kravchenko/correct_rdms_1000_weighted/triangle_corr_weighted.png', '/vol/tcm11/kravchenko/triangle_vs_kagome.png', '/vol/tcm11/kravchenko/triangle_vs_kagome_weighted.png'])

dset=['advertisement', 'art', 'infographics', 'interior_design', 'objects', 'scenes', 'suprematism']
dset_natural=['interior_design', 'objects', 'scenes']
dset_artificial=['advertisement', 'art', 'infographics', 'suprematism']

images_natural=[]
for t in dset_natural:
	i1="mssc_figures/fft_"+t+"_regression.png"
	images_natural.append(i1)
	i2="mssc_figures/fft_"+t+"_regression_2-6.png"
	images_natural.append(i2)

images_artificial=[]
for t in dset_artificial:
	i1="mssc_figures/fft_"+t+"_regression.png"
	images_artificial.append(i1)
	i2="mssc_figures/fft_"+t+"_regression_2-6.png"
	images_artificial.append(i2)

combine_images(columns=2, space=20, images=images_natural, name='mssc_figures/all_natural_sets_regression.png')
combine_images(columns=2, space=20, images=images_artificial, name='mssc_figures/all_artificial_sets_regression.png')



