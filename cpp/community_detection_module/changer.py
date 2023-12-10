import re
import os

def find_and_replace_omp_parallel_for(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Define the regular expression pattern for matching the desired code block
    pattern = re.compile(r'#pragma omp parallel for(.*?)(?://End of for)', re.DOTALL)

    # Find all matches
    matches = pattern.findall(content)

    # Perform modifications for each match
    for match in matches:
        # Replace #pragma omp parallel for with the specified structure
        replacement = (
            '#pragma omp parallel\n'
            '{\n'
            'mgp_track_current_thread_allocations(mg_graph);\n'
            '#pragma omp for\n'
            f'{match}\n'
            'mgp_untrack_current_thread_allocations(mg_graph);\n'
            '}'
        )

        # Replace the entire matched block with the modified content
        content = content.replace(f'#pragma omp parallel for{match}//End of for', replacement)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)

# Specify the directory path
directory_path = 'grappolo/BasicCommunitiesDetection'  # Replace with the actual path to your directory

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.cpp'):  # Modify the condition based on your file extension
        file_path = os.path.join(directory_path, filename)
        find_and_replace_omp_parallel_for(file_path)
