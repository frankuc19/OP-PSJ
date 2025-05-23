# generate_keys.py
import bcrypt

# Pide al usuario que ingrese una contraseña
password_to_hash = input("Ingresa la contraseña que quieres hashear: ").encode('utf-8')

# Genera el hash
hashed_password = bcrypt.hashpw(password_to_hash, bcrypt.gensalt())

# Imprime el hash para que puedas copiarlo y pegarlo en secrets.toml
print(f"\nContraseña hasheada (cópiala en tu archivo secrets.toml):")
print(hashed_password.decode())