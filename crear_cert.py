from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from datetime import datetime, timedelta

# Clave privada
key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

# Datos del certificado
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, u"MX"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"CDMX"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, u"Ciudad de México"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Titanic App"),
    x509.NameAttribute(NameOID.COMMON_NAME, u"titanic.local"),
])

cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.utcnow())
    .not_valid_after(datetime.utcnow() + timedelta(days=365))
    .add_extension(x509.SubjectAlternativeName([x509.DNSName(u"titanic.local")]), critical=False)
    .sign(key, hashes.SHA256())
)

# Guardar archivos
with open("key.pem", "wb") as f:
    f.write(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))

with open("cert.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print("✅ Certificados creados: cert.pem y key.pem")
