import trimesh

def clean_mesh(input_path, output_path):
    # Load the mesh or scene
    mesh = trimesh.load(input_path)

    # If it's a Scene (multiple geometries), combine into one mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump().geometry.values()))

    print("Original mesh stats:")
    print(mesh)

    # Update faces to remove duplicates and degenerates (new way)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())

    # Remove unused vertices
    mesh.remove_unreferenced_vertices()

    # Clean up invalid values
    mesh.remove_infinite_values()

    # Merge close vertices (alternative to deprecated remove_duplicate_vertices)
    mesh.merge_vertices()

    # Fix normals
    mesh.fix_normals()

    # Optional: fill small holes
    mesh.fill_holes()

    # Optional: move to origin
    mesh.rezero()

    # Export cleaned mesh
    mesh.export(output_path)

    print(f"\nCleaned mesh saved to: {output_path}")
    print("Cleaned mesh stats:")
    print(mesh)

# Example usage
if __name__ == "__main__":
    input_file = "kimera_raw_mesh.ply"     # Replace with your actual input path
    output_file = "kimera_cleaned_mesh.ply"
    clean_mesh(input_file, output_file)
