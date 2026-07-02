// Stub rerun.hpp - all operations are no-ops for compilation
#pragma once
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstddef> // for size_t

namespace rerun {

    struct Error {
        void handle() const {}
        static Error ok() { return {}; }
    };

    namespace components {
        struct Translation3D {
            float x = 0, y = 0, z = 0;
            Translation3D() = default;
            Translation3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
            template<typename T>
            Translation3D(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
        };
    }

    struct Quaternion {
        float w = 1, x = 0, y = 0, z = 0;
        static Quaternion from_wxyz(float w_, float x_, float y_, float z_) { return {}; }
    };

    struct Transform3D {
        Transform3D(const components::Translation3D&, const Quaternion&) {}
    };

    struct Pinhole {
        template<typename T>
        Pinhole(const T&) {}
        Pinhole with_resolution(int, int) const { return *this; }
        Pinhole with_resolution(size_t, size_t) const { return *this; }
    };

    struct WidthHeight {
        WidthHeight(int, int) {}
        WidthHeight(size_t, size_t) {}
    };

    template<typename T>
    struct Collection {
        Collection(const std::vector<T>&) {}
        Collection(T*, size_t) {}
    };

    struct Image {
        static Image from_rgb24(const Collection<uint8_t>&, const WidthHeight&) { return {}; }
    };

    struct Points3D {
        Points3D(const Collection<float>&) {}
    };

    class RecordingStream {
    public:
        RecordingStream(const char*) {}
        RecordingStream(const std::string&) {}

        void set_time_seconds(const std::string&, double) const {}
        void set_time_seconds(const char*, double) const {}

        template<typename T>
        void log(const std::string&, const T&) const {}
        template<typename T>
        void log(const char*, const T&) const {}

        bool is_enabled() const { return false; }
    };
}
