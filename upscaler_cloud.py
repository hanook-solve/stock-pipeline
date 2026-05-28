import os
from PIL import Image, ImageFilter

PENDING_FOLDER  = "Pending"
UPSCALED_FOLDER = "Upscaled"

os.makedirs(UPSCALED_FOLDER, exist_ok=True)

valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
images = [f for f in os.listdir(PENDING_FOLDER)
          if f.lower().endswith(valid_ext)]

if not images:
    print("No images found in Pending folder.")
else:
    print(f"Found {len(images)} images.\n")
    success = 0
    failed  = 0

    for i, filename in enumerate(images, 1):
        input_path  = os.path.join(PENDING_FOLDER, filename)
        output_name = os.path.splitext(filename)[0] + '.jpg'
        output_path = os.path.join(UPSCALED_FOLDER, output_name)

        print(f"[{i}/{len(images)}] {filename}")

        try:
            img = Image.open(input_path).convert('RGB')
            new_size = (img.width * 2, img.height * 2)
            img = img.resize(new_size, Image.LANCZOS)
            img = img.filter(ImageFilter.SHARPEN)
            img.save(output_path, format='JPEG',
                     quality=97, dpi=(300, 300), subsampling=0)

            mb = os.path.getsize(output_path) / (1024*1024)
            print(f"  ✓ {output_name} ({mb:.2f} MB)")
            success += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1

    print(f"\nDone! ✓ {success}  ✗ {failed}")