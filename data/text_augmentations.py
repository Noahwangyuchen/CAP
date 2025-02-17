text_augs = {
    'EuroSAT': {

    },
    'RESICS45': {
        'chaparral': 'a dense, patchy, and irregular mosaic of shrubland vegetation, with a dominant brown or beige tone, and a speckled or mottled texture created by the mixture of shrubs, bare soil, and rocky outcrops',
        'terrace': 'a series of flat or gently sloping, rectangular or stepped areas, usually with distinct boundaries and varying tones or textures that distinguish the terraced fields'
    },
    'DTD': {
        'lacelike': 'an intricate pattern of delicate, interwoven threads or shapes resembling filigree, often with a combination of open spaces and detailed designs that convey a sense of elegance and intricacy',
        'gauzy': 'a soft, semi-transparent fabric with delicate, airy layers that gently diffuse light, creating a light and ethereal appearance',
        'veined': 'a network of prominent, branching patterns that resemble the veins of a leaf', 
        'pitted': 'a surface covered in small, rounded or irregular depressions, such as craters, holes, or cavities'
    },
    'Flowers102': {
        'sweet william': 'sweet william features dense, flat-topped clusters of small, frilly-edged blooms in vibrant shades of pink, red, purple, or white, often adorned with intricate contrasting patterns at the center',
        'sweet pea': 'sweet pea features with butterfly-shaped petals in a range of soft colors, including pink, lavender, purple, and white, with a sweet fragrance and a climbing vine that often supports their vibrant blooms',
        'globe flower': 'globe flower is distinguished by bright yellow, orange, or white blooms that are rounded and globe-shaped, with 5-10 petal-like sepals that curve outward to form a ball-like shape, often measuring 2-5 inches in diameter, and borne on tall, slender stems with deeply lobed leaves',
        'love in the mist': 'love in the mist is distinguished by intricate, airy appearance, with delicate, spidery, fern-like foliage surrounding vibrant, star-shaped blooms in shades of blue, purple, or white, often accented with a halo of fine, wispy tendrils',
        'great masterwort': 'great masterwort features with umbrella-like clusters of small, star-shaped blooms surrounded by showy, papery bracts in soft shades of white, pink, or green, creating a delicate yet structured appearance atop slender stems',
        'buttercup': 'buttercup features with cup-shaped petals in bright yellow, often with a simple, cheerful appearance and finely divided, green leaves at the base',
        'bishop of llandaff': 'the bishop of llandaff dahlia is notable for its vivid, velvety scarlet blooms with a simple, open-petaled structure, set against striking dark, almost black, foliage that provides dramatic contrast'
    },
    'CUB': {
        'blue winged warbler': 'a blue-winged warbler is a small songbird with striking yellow plumage, a black eye line, a bluish-gray wing with two white wing bars, and a slender black bill',
        'sayornis': 'sayornis, commonly known as the tyrant flycatcher, typically has a grayish or olive-brown body, with a slightly darker head, pale underparts, and a distinctive white belly, often complemented by a subtle contrast between its wings, which are dark with light edges, and its tail, which is often slightly notched',
        'scott oriole': 'scott oriole is distinguished by its striking black and yellow plumage, with a deep black head, back, and chest contrasting with bright yellow underparts and a vibrant yellow rump, along with white wing bars',
        'wilson warbler': 'wilson warbler is a small, bright yellow bird with a distinctive black cap on top of its head, white eye rings, and a greenish yellow back and wings, making it a visually striking species among North American warblers.',
        'orange crowned warbler': 'orange crowned warbler is characterized by its olive green upperparts, a subtle orange patch on the crown that is often hidden, and a pale, yellowish underbelly with distinct dark streaks on its sides'
    },
    'FGVCAircraft': {
        '767-400': 'Boeing 767-400 is distinguishable by its long, slender fuselage, distinctive raked wingtips, and a relatively small tail section compared to its overall length, with a characteristic Boeing 767 nose and cockpit shape',
        '757-200': 'Boeing 757-200 is distinguished by its sleek, narrow-body fuselage, high T-shaped tail, elongated nose, and tall, slender landing gear, complemented by its dual turbofan engines mounted below the wings.'
    },
}

def get_text_aug(dataset):
    return text_augs.get(dataset, {})