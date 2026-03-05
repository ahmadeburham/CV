import easyocr


def extract_text(image_path):

    reader = easyocr.Reader(["ar", "en"])

    results = reader.readtext(image_path)

    extracted_text = []

    for r in results:
        extracted_text.append(r[1])

    return extracted_text