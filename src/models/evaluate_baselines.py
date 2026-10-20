import numpy as np

DATA = "data/processed/trajectory_sequences.npz"
data = np.load(DATA)

X = data["X_test"]
Y = data["y_test"]

def haversine_km(actual, predicted):
    lat1, lon1 = np.radians(actual[..., 0]), np.radians(actual[..., 1])
    lat2, lon2 = np.radians(predicted[..., 0]), np.radians(predicted[..., 1])

    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def predict_course_speed(x):
    lat = x[:, -1, 0]
    lon = x[:, -1, 1]
    sog = x[:, -1, 2]
    sin_cog = x[:, -1, 3]
    cos_cog = x[:, -1, 4]

    course = np.arctan2(sin_cog, cos_cog)
    speed = sog * 1.852 / 60

    predictions = []

    for minutes in [5, 10, 15, 20, 25]:
        distance = speed * minutes
        lat1 = np.radians(lat)
        lon1 = np.radians(lon)
        bearing = course
        angular_distance = distance / 6371

        lat2 = np.arcsin(
            np.sin(lat1) * np.cos(angular_distance)
            + np.cos(lat1) * np.sin(angular_distance) * np.cos(bearing)
        )

        lon2 = lon1 + np.arctan2(
            np.sin(bearing) * np.sin(angular_distance) * np.cos(lat1),
            np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2)
        )

        predictions.append(np.column_stack([
            np.degrees(lat2),
            (np.degrees(lon2) + 540) % 360 - 180
        ]))

    return np.stack(predictions, axis=1)

course_speed_predictions = predict_course_speed(X)
stationary_predictions = np.repeat(X[:, -1:, :2], 5, axis=1)

for name, predictions in [
    ("Course-speed baseline", course_speed_predictions),
    ("Stationary baseline", stationary_predictions)
]:
    errors = haversine_km(Y, predictions)

    print(f"\n{name}:")
    for i, minutes in enumerate([5, 10, 15, 20, 25]):
        print(f"{minutes} min: {errors[:, i].mean():.3f} km")

    print(f"Overall mean: {errors.mean():.3f} km")