#include "gui/win32/app_icons.h"

#include <cstring>
#include <vector>

#include "gui/win32/resources/icon_bmp.h"
#include "gui/win32/resources/logo_bmp.h"

namespace {

// Builds an HBITMAP of an embedded BMP file image (logo_bmp.h/icon_bmp.h)
// scaled to `w` x `h`. The caller owns it and must DeleteObject() it. HALFTONE
// stretch mode so heavily downscaled artwork averages rather than drops pixels.
HBITMAP makeBitmapFromBmp(const unsigned char* bmp, int w, int h) {
    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;
    memcpy(&fileHeader, bmp, sizeof(fileHeader));
    memcpy(&infoHeader, bmp + sizeof(fileHeader), sizeof(infoHeader));
    const unsigned char* bits = bmp + fileHeader.bfOffBits;

    HDC screen = GetDC(nullptr);
    HDC mem = CreateCompatibleDC(screen);
    HBITMAP bitmap = CreateCompatibleBitmap(screen, w, h);
    HGDIOBJ old = SelectObject(mem, bitmap);
    SetStretchBltMode(mem, HALFTONE);
    SetBrushOrgEx(mem, 0, 0, nullptr);
    StretchDIBits(mem, 0, 0, w, h, 0, 0, infoHeader.biWidth, infoHeader.biHeight, bits,
                  reinterpret_cast<const BITMAPINFO*>(bmp + sizeof(fileHeader)), DIB_RGB_COLORS, SRCCOPY);
    SelectObject(mem, old);
    DeleteDC(mem);
    ReleaseDC(nullptr, screen);
    return bitmap;
}

// The application icon (icon_bmp.h) at `size` x `size` pixels - Windows asks
// for different sizes in different places (title bar/tray vs. taskbar/Alt-Tab,
// see SM_CXSMICON/SM_CXICON, and again at other DPI scalings). Each embedded
// size is rendered separately for crispness, so this uses the smallest one
// that is at least `size` (or the largest, if none is) - an exact match is
// copied 1:1 with no resampling at all, only an in-between request (e.g. 20
// or 40) gets scaled, and only slightly. Fully opaque (an all-zero AND mask).
// Never freed: a handful of icons for the life of the process.
HICON makeAppIcon(int size) {
    const IconImage* image = &kIconImages[0];
    for (const IconImage& candidate : kIconImages) {
        image = &candidate;
        if (candidate.size >= size)
            break;
    }
    HBITMAP color = makeBitmapFromBmp(image->bmp, size, size);
    std::vector<unsigned char> maskBits(((size + 15) / 16) * 2 * size, 0); // 1bpp rows are word-aligned
    HBITMAP mask = CreateBitmap(size, size, 1, 1, maskBits.data());
    ICONINFO info{};
    info.fIcon = TRUE;
    info.hbmMask = mask;
    info.hbmColor = color;
    HICON icon = CreateIconIndirect(&info);
    DeleteObject(mask);
    DeleteObject(color);
    return icon;
}

} // namespace

namespace gui {

HBITMAP makeLogoBitmap(int w, int h) {
    return makeBitmapFromBmp(kLogoBmp, w, h);
}

HICON appIconLarge() {
    static HICON icon = makeAppIcon(GetSystemMetrics(SM_CXICON));
    return icon;
}

HICON appIconSmall() {
    static HICON icon = makeAppIcon(GetSystemMetrics(SM_CXSMICON));
    return icon;
}

} // namespace gui
