from pathlib import Path
import subprocess
import tempfile
import shutil
import sys
import os

import gradio as gr


def buildInferenceCommand(
    pythonExecutable: Path,
    repoRoot: Path,
    inputPath: Path,
    outputDir: Path,
    modelName: str,
    outscale: float,
) -> list[str]:
    return [
        str(pythonExecutable),
        str(repoRoot / "inference_realesrgan.py"),
        "-n",
        modelName,
        "-i",
        str(inputPath),
        "-o",
        str(outputDir),
        "--outscale",
        str(outscale),
    ]


def getCurrentPythonExecutable() -> Path:
    return Path(sys.executable)


def getServerPort(environment: dict[str, str]) -> int:
    serverPort = environment.get("GRADIO_SERVER_PORT", "7860")
    return int(serverPort)


def getOutputPath(outputDir: Path, inputStem: str) -> Path:
    candidates = sorted(outputDir.glob(f"{inputStem}_out.*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return outputDir / f"{inputStem}_out.png"


def upscaleImage(inputImagePath: str, modelName: str, outscale: float) -> str:
    repoRoot = Path(__file__).resolve().parent
    outputDir = repoRoot / "resultsUi"
    outputDir.mkdir(parents=True, exist_ok=True)
    tempInputDir = Path(tempfile.mkdtemp(prefix="realesrgan-ui-"))
    inputPath = Path(inputImagePath)
    tempInputPath = tempInputDir / inputPath.name
    shutil.copy2(inputPath, tempInputPath)

    command = buildInferenceCommand(
        pythonExecutable=getCurrentPythonExecutable(),
        repoRoot=repoRoot,
        inputPath=tempInputPath,
        outputDir=outputDir,
        modelName=modelName,
        outscale=outscale,
    )

    try:
        subprocess.run(command, check=True, cwd=repoRoot)
    finally:
        shutil.rmtree(tempInputDir, ignore_errors=True)

    outputPath = getOutputPath(outputDir, inputPath.stem)
    if not outputPath.exists():
        raise RuntimeError("Upscaled image was not generated")
    return str(outputPath)


def createInterface() -> gr.Blocks:
    with gr.Blocks(title="Real-ESRGAN Upscale UI") as app:
        gr.Markdown("## Real-ESRGAN Upscale")
        with gr.Row():
            inputImage = gr.Image(type="filepath", label="Input image")
            outputImage = gr.Image(type="filepath", label="Upscaled image")
        with gr.Row():
            modelName = gr.Dropdown(
                choices=["RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B"],
                value="RealESRGAN_x4plus",
                label="Model",
            )
            outscale = gr.Slider(minimum=1, maximum=4, value=2, step=0.5, label="Output scale")
        runButton = gr.Button("Upscale")
        runButton.click(fn=upscaleImage, inputs=[inputImage, modelName, outscale], outputs=[outputImage])
    return app


if __name__ == "__main__":
    createInterface().launch(server_name="127.0.0.1", server_port=getServerPort(dict(os.environ)))
