import { NextRequest, NextResponse } from 'next/server';
import { writeFile } from 'fs/promises';
import { join } from 'path';

export async function POST(request: NextRequest) {
  try {
    const data = await request.formData();
    const file: File | null = data.get('video') as unknown as File;

    if (!file) {
      return NextResponse.json({ success: false, error: 'No file uploaded' }, { status: 400 });
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Ensure the uploads directory exists
    const uploadsDir = join(process.cwd(), 'uploads');
    await import('fs').then(fs => fs.promises.mkdir(uploadsDir, { recursive: true }));

    const path = join(uploadsDir, file.name);
    await writeFile(path, buffer);

    console.log('File saved successfully:', file.name);  // Add this log

    return NextResponse.json({ success: true, filename: file.name });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json({ success: false, error: 'Upload failed' }, { status: 500 });
  }
}