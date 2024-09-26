import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(request: NextRequest) {
  const url = new URL(request.url);
  const fileParam = url.searchParams.get('file');
  console.log('Requested file:', fileParam);
  
  if (!fileParam) {
    console.log('No file specified');
    return NextResponse.json({ error: 'No file specified' }, { status: 400 });
  }

  const uploadsDir = path.join(process.cwd(), 'uploads');
  const filePath = path.join(uploadsDir, fileParam);
  
  console.log('Full file path:', filePath);

  if (!fs.existsSync(filePath)) {
    console.error(`File not found: ${filePath}`);
    return NextResponse.json({ error: 'File not found' }, { status: 404 });
  }

  try {
    const fileBuffer = fs.readFileSync(filePath);
    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'text/vtt',
        'Content-Disposition': `attachment; filename="${fileParam}"`,
      }
    });
  } catch (error) {
    console.error(`Error reading file: ${error}`);
    return NextResponse.json({ error: 'Error reading file' }, { status: 500 });
  }
}