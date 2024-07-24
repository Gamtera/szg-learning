import random

shapes = {
    "dikdörtgen": """
    -----
    |   |
    |   |
    -----
    """,
    "daire": """
     ***
    *   *
    *   *
     ***
    """,
    "üçgen": """
      /\\
     /  \\
    /____\\
    """,
    "kare": """
    --------
    |      |
    |      |
    --------
    """
}

# Rastgele bir şekil seç
selected_shape = random.choice(list(shapes.keys()))

# Seçilen şekli çiz
print(shapes[selected_shape])

# Kullanıcıdan tahmin al
cisim = input(f"Bu şekil nedir? (Seçenekler: {', '.join(shapes.keys())}): ")

if cisim.lower() == selected_shape:
    print(f"Tebrikler! Bu bir {selected_shape}.")
else:
    print(f"Üzgünüm, bu bir {selected_shape}.")