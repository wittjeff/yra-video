import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import util from "util";
import fs from "fs";
import path from "path";

const execPromise = util.promisify(exec);

export async function POST(request: NextRequest) {
  const { filename } = await request.json();

  try {
    const inputPath = path.join(process.cwd(), "uploads", filename);
    const outputAudioPath = path.join(
      process.cwd(),
      "uploads",
      `${filename}.wav`
    );

    // Use normalized paths for Windows
    const normalizedInputPath = inputPath.replace(/\\/g, "/");
    const normalizedOutputAudioPath = outputAudioPath.replace(/\\/g, "/");

    // Extract audio
    const ffmpegCommand = `ffmpeg -i "${normalizedInputPath}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "${normalizedOutputAudioPath}"`;
    console.log("Executing FFmpeg command:", ffmpegCommand); // Add this log
    await execPromise(ffmpegCommand);

    // TODO: Implement speech-to-text processing here
    //const whisperCommand = `python -m whisper "${normalizedInputPath}" --model tiny --device cuda --output_dir "${path.dirname(normalizedInputPath)}"`;
    const whisperCommand = `"C:/Users/moham/AppData/Local/Programs/Python/Python310/python.exe" -m whisper "${normalizedInputPath}" --model tiny --device cuda --output_dir "${path.dirname(
      normalizedInputPath
    )}"`;
    console.log("Executing Whisper command:", whisperCommand);
    const { stdout, stderr } = await execPromise(whisperCommand);

    console.log("Whisper stdout:", stdout);
    console.log("Whisper stderr:", stderr);

    // For this example, we'll just create a dummy caption file
    //const captions = 'This is a dummy caption file.\nReplace with actual captions.';
    const captionPath = `${normalizedInputPath}.txt`;
    //const captionPath = path.join(process.cwd(), 'uploads', `${filename}.txt`);
    //fs.writeFileSync(captionPath, captions);

    //return NextResponse.json({ success: true, captionUrl: `/api/download/route?file=${filename}.txt` });

    const vttPath = `${normalizedInputPath.slice(0, -4)}.vtt`;
    let vttContent = "";
    if (fs.existsSync(vttPath)) {
      vttContent = fs.readFileSync(vttPath, "utf-8");
    } else {
      console.error("VTT file not found:", vttPath);
    }

    const vttFilename = path.basename(vttPath);
    return NextResponse.json({
      success: true,
      captionUrl: `/api/download?file=${encodeURIComponent(vttFilename)}`,
      vttContent: vttContent,
    });
  } catch (error) {
    console.error("Error processing video:", error);
    return NextResponse.json(
      { success: false, error: "Error processing video" },
      { status: 500 }
    );
  }
}
