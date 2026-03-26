from pathlib import Path
import sys
import tempfile

from app import buildInferenceCommand, getCurrentPythonExecutable, getOutputPath, getServerPort


def testBuildInferenceCommandUsesModelScaleAndOutput():
    command = buildInferenceCommand(
        pythonExecutable=Path("C:/proj/.venv/Scripts/python.exe"),
        repoRoot=Path("C:/proj"),
        inputPath=Path("C:/images/input.png"),
        outputDir=Path("C:/proj/resultsUi"),
        modelName="RealESRNet_x4plus",
        outscale=2.0,
    )

    assert Path(command[0]).name == "python.exe"
    assert Path(command[1]).name == "inference_realesrgan.py"
    assert "-n" in command and "RealESRNet_x4plus" in command
    assert "--outscale" in command and "2.0" in command
    assert "-o" in command
    assert Path(command[command.index("-o") + 1]).name == "resultsUi"


def testGetCurrentPythonExecutableUsesRunningInterpreter():
    executablePath = getCurrentPythonExecutable()
    assert executablePath == Path(sys.executable)


def testGetServerPortDefaultsTo7860WhenUnset():
    assert getServerPort({}) == 7860


def testGetServerPortReadsEnvironmentValue():
    assert getServerPort({"GRADIO_SERVER_PORT": "7861"}) == 7861


def testGetOutputPathFindsJpgResult():
    with tempfile.TemporaryDirectory() as tempDir:
        outputDir = Path(tempDir)
        expected = outputDir / "photo_2022-09-24_23-27-20_out.jpg"
        expected.touch()
        outputPath = getOutputPath(outputDir, "photo_2022-09-24_23-27-20")
        assert outputPath == expected


def testGetOutputPathFallsBackToPngResult():
    outputDir = Path("C:/proj/resultsUi")
    outputPath = getOutputPath(outputDir, "sample")
    expected = outputDir / "sample_out.png"
    assert outputPath == expected
