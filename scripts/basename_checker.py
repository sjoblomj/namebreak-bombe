#!/usr/bin/env python3
import sys
import argparse
import struct
import math

CRYPT_TABLE = [0] * 0x500

def prepare_crypt_table():
    seed = 0x00100001
    for i in range(0x100):
        index = i
        for _ in range(5):
            seed = (seed * 125 + 3) % 0x2AAAAB
            temp1 = (seed & 0xFFFF) << 16
            seed = (seed * 125 + 3) % 0x2AAAAB
            temp2 = seed & 0xFFFF
            CRYPT_TABLE[index] = temp1 | temp2
            index += 0x100

prepare_crypt_table()

def hash_string(string, hash_type):
    """Calculates MPQ Hash for a string. String should be uppercase."""
    seed1 = 0x7FED7FED
    seed2 = 0xEEEEEEEE
    for char in string.upper():
        ch = ord(char)
        seed1 = CRYPT_TABLE[(hash_type << 8) + ch] ^ (seed1 + seed2)
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3
    return seed1 & 0xFFFFFFFF

def find_mpq_header(file_obj):
    """Scans file on 512-byte boundaries to locate an embedded MPQ header."""
    file_obj.seek(0, 2)
    file_length = file_obj.tell()
    file_obj.seek(0)

    offset = 0
    while offset < file_length:
        file_obj.seek(offset)
        magic = file_obj.read(4)
        if magic == b'MPQ\x1a':
            return offset
        offset += 0x200
    return None

def decrypt_block(data, key):
    """Decrypts MPQ-encrypted data with the given key."""
    out = bytearray(data)
    seed1, seed2 = key, 0xEEEEEEEE
    for i in range(0, len(out) - len(out) % 4, 4):
        seed2 = (seed2 + CRYPT_TABLE[0x400 + (seed1 & 0xFF)]) & 0xFFFFFFFF
        cipher = struct.unpack('<I', out[i:i+4])[0]
        plain = cipher ^ ((seed1 + seed2) & 0xFFFFFFFF)
        plain &= 0xFFFFFFFF
        seed1 = (((~seed1 << 21) & 0xFFFFFFFF) + 0x11111111) | (seed1 >> 11)
        seed1 &= 0xFFFFFFFF
        seed2 = (plain + seed2 + (seed2 << 5) + 3) & 0xFFFFFFFF
        out[i:i+4] = struct.pack('<I', plain)
    return out

def recover_encryption_key_from_sector_table(sector_table, p0, compressed_size, file_size, block_offset, flags):
    """Recovers HashType 3 (encryption_key) by solving for the key using known plaintext P0."""
    c0 = struct.unpack('<I', sector_table[:4])[0]

    # Check all 256 possible low bytes for seed1. More than one low byte can solve
    # for P0, so a candidate is only taken if it decrypts the whole sector table
    # into ascending offsets that end inside the file's compressed data.
    for i in range(256):
        t_val = CRYPT_TABLE[0x400 + i]
        k = (c0 ^ p0) - 0xEEEEEEEE - t_val
        k &= 0xFFFFFFFF

        if (k & 0xFF) != i:
            continue

        offsets = struct.unpack(f'<{len(sector_table) // 4}I', decrypt_block(sector_table, k))
        if any(a > b for a, b in zip(offsets, offsets[1:])) or offsets[-1] > compressed_size:
            continue

        file_key = (k + 1) & 0xFFFFFFFF

        if flags & 0x00020000:  # MPQ_FILE_FIX_KEY
            encryption_key = ((file_key ^ file_size) - block_offset) & 0xFFFFFFFF
        else:
            encryption_key = file_key

        return encryption_key

    return None

def extract_mpq_encryption_key(archive_path, target_hash_a, target_hash_b):
    """Parses MPQ archive and recovers the encryption_key for a target file."""
    with open(archive_path, 'rb') as f:
        mpq_offset = find_mpq_header(f)
        if mpq_offset is None:
            print("Error: No valid MPQ header found in file.")
            sys.exit(1)

        f.seek(mpq_offset)
        header_data = f.read(32)

        header_size, archive_size, format_version, block_size_shift, hash_table_offset, block_table_offset, hash_table_entries, block_table_entries = struct.unpack('<IIHHIIII', header_data[4:32])
        sector_size = 512 << block_size_shift

        # Read and Decrypt Hash Table
        f.seek(mpq_offset + hash_table_offset)
        hash_table_raw = decrypt_block(f.read(hash_table_entries * 16), hash_string("(hash table)", 3))

        # Locate block index for Hash A and Hash B
        block_index = None
        for i in range(hash_table_entries):
            h_a, h_b, lc, platform, b_idx = struct.unpack('<IIHHI', hash_table_raw[i*16:(i+1)*16])
            if h_a == target_hash_a and h_b == target_hash_b and b_idx != 0xFFFFFFFF:
                block_index = b_idx
                break

        if block_index is None:
            print("Error: File with specified Hash A/B not found in archive.")
            sys.exit(1)

        # Read and Decrypt Block Table
        f.seek(mpq_offset + block_table_offset)
        block_table_raw = decrypt_block(f.read(block_table_entries * 16), hash_string("(block table)", 3))

        block_offset, compressed_size, file_size, flags = struct.unpack('<IIII', block_table_raw[block_index*16:(block_index+1)*16])

        if not (flags & 0x00010000):  # MPQ_FILE_ENCRYPTED
            print("Error: Targeted file is not encrypted.")
            sys.exit(1)

        # Determine P0
        num_sectors = math.ceil(file_size / sector_size)
        has_crc = bool(flags & 0x04000000)
        p0 = (num_sectors + 2 if has_crc else num_sectors + 1) * 4

        # Read the file's encrypted sector table, which is P0 bytes long
        f.seek(mpq_offset + block_offset)
        sector_table = f.read(p0)

        return recover_encryption_key_from_sector_table(sector_table, p0, compressed_size, file_size, block_offset, flags)

def auto_int(x):
    return int(x, 0)

def main():
    parser = argparse.ArgumentParser(description="Verify MPQ bare filename against encryption key (HashType 3).")
    parser.add_argument("bare_filename", nargs="?", help="The filename guess to check (without directory)")
    parser.add_argument("-a", "--archive", help="Path to MPQ archive file")
    parser.add_argument("--hash-a", type=auto_int, help="Target file Hash A (hex or decimal)")
    parser.add_argument("--hash-b", type=auto_int, help="Target file Hash B (hex or decimal)")
    parser.add_argument("-k", "--key", type=auto_int, help="Encryption key (HashType 3, hex or decimal)")
    parser.add_argument("--print-key", action="store_true", help="Print the encryption key (HashType 3) and exit: of bare_filename, or with --archive, of the file at --hash-a/--hash-b")

    args = parser.parse_args()

    has_archive_info = args.archive and args.hash_a is not None and args.hash_b is not None

    if args.print_key:
        if args.archive:
            if not has_archive_info:
                parser.error("--print-key with --archive requires --hash-a and --hash-b.")
            if args.bare_filename is not None:
                parser.error("--print-key with --archive prints the archive file's key; do not supply bare_filename.")
            recovered_key = extract_mpq_encryption_key(args.archive, args.hash_a, args.hash_b)
            if recovered_key is None:
                print("Error: Failed to recover encryption key from sector table.")
                sys.exit(1)
            print(f"{recovered_key:08x}")
        else:
            if args.bare_filename is None:
                parser.error("--print-key requires bare_filename, or --archive, --hash-a and --hash-b.")
            print(f"{hash_string(args.bare_filename, 3):08x}")
        sys.exit(0)

    if args.bare_filename is None:
        parser.error("bare_filename is required unless using --print-key with --archive.")

    if not has_archive_info and args.key is None:
        parser.error("You must supply either --key/-k, full MPQ parameters (--archive, --hash-a, --hash-b) or --print-key.")

    target_encryption_key = None

    if has_archive_info:
        recovered_key = extract_mpq_encryption_key(args.archive, args.hash_a, args.hash_b)
        if recovered_key is None:
            print("Error: Failed to recover encryption key from sector table.")
            sys.exit(1)

        if args.key is not None:
            # Explicitly verify user-supplied key matches recovered key
            if args.key != recovered_key:
                print(f"Key mismatch: Supplied key (0x{args.key:08X}) does not match MPQ recovered key (0x{recovered_key:08X}).")
                print("Does not match")
                sys.exit(1)

        target_encryption_key = recovered_key
    else:
        # Take supplied key at face value
        target_encryption_key = args.key

    # Check bare filename match
    guessed_key = hash_string(args.bare_filename, 3)

    if guessed_key == target_encryption_key:
        print("Matches")
        sys.exit(0)
    else:
        print("Does not match")
        sys.exit(1)

if __name__ == "__main__":
    main()
