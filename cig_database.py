class Cigarette:
    def __init__(self, name, variations):

        self.name = name
        self.variations  = []
        self.color_variations = set()
        self.images = set()
        self.loc = set()

        for variation in variations:
            size = variation["size"]
            color = variation["color"]
            image = variation["image"]
            loc = f"images/src/{image}"

            self.variations.append({"size": size, "color": color, "image": image})
            self.color_variations .add(color)
            self.images.add(image)
            self.loc.add(loc)

def __str__(self):
        return f"{self.name} - Variations: {len(self.variations)}, Color Variations: {len(self.color_variations )}, Images: {len(self.images)} , loc: {self.loc}"

# Cigarette data
product_data = {
    "Seneca": {
        "variations": [
            {"size": "Kings", "color": "Red", "image": "Seneca_Kings_red.jpg"},
            {"size": "Kings", "color": "Blue", "image": "Seneca_Kings_blue.jpg"},
            {"size": "Kings", "color": "Green", "image": "Seneca_Kings_green.jpg"},
            {"size": "Kings", "color": "LGreen", "image": "Seneca_Kings_lgreen.jpg"},
            {"size": "Kings", "color": "ELGreen", "image": "Seneca_Kings_elgreen.jpg"},
            {"size": "Kings", "color": "Medium", "image": "Seneca_Kings_medium.jpg"},
            {"size": "Kings", "color": "Silver", "image": "Seneca_Kings_silver.jpg"},
            {"size": "Kings", "color": "Non-Filter", "image": "Seneca_Kings_nonfilter.jpg"},
            {"size": "100s", "color": "Red", "image": "Seneca_100s_red.jpg"},
            {"size": "100s", "color": "Blue", "image": "Seneca_100s_blue.jpg"},
            {"size": "100s", "color": "Green", "image": "Seneca_100s_green.jpg"},
            {"size": "100s", "color": "LGreen", "image": "Seneca_100s_lgreen.jpg"},
            {"size": "100s", "color": "ELGreen", "image": "Seneca_100s_elgreen.jpg"},
            {"size": "100s", "color": "Medium", "image": "Seneca_100s_medium.jpg"},
            {"size": "100s", "color": "Silver", "image": "Seneca_100s_silver.jpg"},
            # Add more variations as needed
        ],
    },
    "247": {
        "variations": [
            {"size": "Kings", "color": "Red", "image": "247_Kings_red.jpg"},
            {"size": "Kings", "color": "Blue", "image": "247_Kings_blue.jpg"},
            {"size": "Kings", "color": "Green", "image": "247_Kings_green.jpg"},
            {"size": "Kings", "color": "Silver", "image": "247_Kings_silver.jpg"},
            {"size": "Kings", "color": "Non-Filter", "image": "247_Kings_nonfilter.jpg"},
            {"size": "100s", "color": "Red", "image": "247_100s_red.jpg"},
            {"size": "100s", "color": "Blue", "image": "247_100s_blue.jpg"},
            {"size": "100s", "color": "Green", "image": "247_100s_green.jpg"},
            {"size": "100s", "color": "Silver", "image": "247_100s_silver.jpg"},
            # Add more variations as needed
        ],
    },
    "Edgefield": {
        "variations": [
            {"size": "Kings", "color": "Red", "image": "Edgefield_Kings_red.jpg"},
            {"size": "Kings", "color": "Blue", "image": "Edgefield_Kings_blue.jpg"},
            {"size": "Kings", "color": "Green", "image": "Edgefield_Kings_green.jpg"},
            {"size": "Kings", "color": "LGreen", "image": "Edgefield_Kings_lgreen.jpg"},   
            {"size": "Kings", "color": "Silver", "image": "Edgefield_Kings_silver.jpg"},
            {"size": "Kings", "color": "Non-Filter", "image": "Edgefield_Kings_nonfilter.jpg"},

            {"size": "100s", "color": "Red", "image": "Edgefield_100s_red.jpg"},
            {"size": "100s", "color": "Blue", "image": "Edgefield_100s_blue.jpg"},
            {"size": "100s", "color": "Green", "image": "Edgefield_100s_green.jpg"},
            {"size": "100s", "color": "LGreen", "image": "Edgefield_100s_lgreen.jpg"},
            {"size": "100s", "color": "Silver", "image": "Edgefield_100s_silver.jpg"},
        ],
    },
    "LD": {
        "variations": [
            {"size": "Kings", "color": "Red", "image": "LD_Kings_red.jpg"},
            {"size": "Kings", "color": "Blue", "image": "LD_Kings_blue.jpg"},
            {"size": "Kings", "color": "Green", "image": "LD_Kings_green.jpg"},
            {"size": "Kings", "color": "LGreen", "image": "LD_Kings_lgreen.jpg"},   
            {"size": "Kings", "color": "Silver", "image": "LD_Kings_silver.jpg"},
            
            {"size": "100s", "color": "Red", "image": "LD_100s_red.jpg"},
            {"size": "100s", "color": "Blue", "image": "LD_100s_blue.jpg"},
            {"size": "100s", "color": "Green", "image": "LD_100s_green.jpg"},
            {"size": "100s", "color": "LGreen", "image": "LD_100s_lgreen.jpg"},
            {"size": "100s", "color": "Silver", "image": "LD_100s_silver.jpg"},
            # Add more variations as needed
        ],
    },
    "Montego": {
        "variations": [
            {"size": "Kings", "color": "Red", "image": "Montego_Kings_red.jpg"},
            {"size": "Kings", "color": "Blue", "image": "Montego_Kings_blue.jpg"},
            {"size": "Kings", "color": "Green", "image": "Montego_Kings_green.jpg"},
            {"size": "Kings", "color": "LGreen", "image": "Montego_Kings_lgreen.jpg"},   
            {"size": "Kings", "color": "Silver", "image": "Montego_Kings_silver.jpg"},
            
            {"size": "100s", "color": "Red", "image": "Montego_100s_red.jpg"},
            {"size": "100s", "color": "Blue", "image": "Montego_100s_blue.jpg"},
            {"size": "100s", "color": "Green", "image": "Montego_100s_green.jpg"},
            {"size": "100s", "color": "LGreen", "image": "Montego_100s_lgreen.jpg"},
            {"size": "100s", "color": "Silver", "image": "Montego_100s_silver.jpg"},
            # Add more variations as needed
        ],
    },
    # Add more products as needed
}


# Create products
products = {}
for product_name, product_info in product_data.items():
    product = Cigarette(product_name, product_info["variations"])
    products[product_name] = product

# Print the inventory
for product_name, product in products.items():
    print(product)

# Accessing a specific product and its size and color variations
selected_product = "Edgefield"
print(f"\nSelected Product: {selected_product}")
print("Variations:", products[selected_product].variations)
print("Color Variations:", products[selected_product].color_variations)
print("Images:", products[selected_product].images)
print("Location:", products[selected_product].loc)
print("\n")
