import random
import string

chars = string.ascii_letters + string.digits + "!@#$%&*"
password = ''.join(random.choices(chars, k=12))
print(f"Password: {password}")