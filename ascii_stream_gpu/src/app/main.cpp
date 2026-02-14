#include "App.h"

#include <Windows.h>
#include <string>
#include <cstring>

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int)
{
    try
    {
        App app;
        app.Initialize(hInst);

        MSG msg{};
        while (app.Running())
        {
            app.Tick();
        }

        app.Shutdown();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::wstring wmsg = L"Fatal error: ";
        wmsg += std::wstring(e.what(), e.what() + std::strlen(e.what()));
        MessageBox(nullptr, wmsg.c_str(), L"Exception", MB_OK | MB_ICONERROR);
        return 0;
    }
}
