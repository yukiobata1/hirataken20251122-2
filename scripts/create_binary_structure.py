import random
import sys

def create_binary_structure(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    atoms = []
    
    reading_atoms = False
    reading_masses = False
    
    for line in lines:
        if "atom types" in line:
            new_lines.append(line.replace("1 atom types", "2 atom types"))
            continue
            
        if "Masses" in line:
            reading_masses = True
            new_lines.append(line)
            continue
            
        if "Atoms" in line:
            reading_masses = False
            reading_atoms = True
            new_lines.append(line)
            continue

        if reading_masses:
            if not line.strip():
                new_lines.append(line)
                continue
            
            # This is a mass line
            parts = line.split()
            if len(parts) >= 2 and parts[0] == '1':
                new_lines.append(line)
                # Add mass 2 immediately after mass 1
                # Ensure we match the indentation/format if possible, or just standard
                new_lines.append(f"{line.replace('1', '2', 1).strip()} # Ga2\n")
            else:
                # Could be comment or other
                new_lines.append(line)
            continue

        if reading_atoms:
            if not line.strip():
                new_lines.append(line)
                continue
            # Atom line
            atoms.append(line)
            continue
            
        # Header or other sections
        new_lines.append(line)

    # Process atoms
    total_atoms = len(atoms)
    indices = list(range(total_atoms))
    random.shuffle(indices)
    type2_indices = set(indices[:total_atoms // 2])

    for i, atom_line in enumerate(atoms):
        parts = atom_line.split()
        # Format: id type x y z
        atom_id = parts[0]
        # original_type = parts[1] 
        
        new_type = '2' if i in type2_indices else '1'
        
        # Reconstruct line
        # The original file might have 5 columns (id type x y z) or more (image flags)
        # We just replace the second column
        parts[1] = new_type
        new_line = " ".join(parts) + "\n"
        new_lines.append(new_line)

    with open(output_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Created {output_file} with {total_atoms} atoms (50% Type 1, 50% Type 2)")

if __name__ == "__main__":
    create_binary_structure('inputs/data.ga_1000', 'inputs/data.ga_binary')