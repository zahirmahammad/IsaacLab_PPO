ffmpeg -i videos/A1FlatWalk.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -c:v gif videos/A1FlatWalk.gif
ffmpeg -i videos/DigitFlatWalk.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -c:v gif videos/DigitFlatWalk.gif
ffmpeg -i videos/FrankaOpenDrawer.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -c:v gif videos/FrankaOpenDrawer.gif
ffmpeg -i videos/G1FlatWalk.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -c:v gif videos/G1FlatWalk.gif
ffmpeg -i videos/PandaLift.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -c:v gif videos/PandaLift.gif
ffmpeg -i videos/SpotFlat.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -c:v gif videos/SpotFlat.gif